//! Waveshare 7.5" V2 (800x480, UC8179 / GD7965) with deep-sleep-safe partial refresh.
//!
//! Drop-in replacement for `src/epd7in5_v2/mod.rs` on top of the m5r branch
//! (`feat-7in5-v2-partial-frame`, it provides the async `DisplayInterface` helpers and
//! `BusyTimeoutError`), plus `RefreshLut::PartialRefresh` from dexterdy's `traits.rs`.
//!
//! # Design
//!
//! * **One data polarity everywhere.** CDI is always written with DDX=01, which makes
//!   bit 1 = white in *both* DTM1 (OLD) and DTM2 (NEW). The framebuffer is sent raw,
//!   OLD and NEW are directly comparable, and N2OCP=1 makes the controller copy NEW→OLD
//!   after every refresh. (Waveshare's C demo instead flips DDX between 0x10 and 0xA9 and
//!   relies on the side effect that a raw DTM1 means "inverse image" under DDX=00 and
//!   "previous image" under DDX=01. Mixing those two worlds is the classic source of
//!   inverted/garbled partial updates.)
//! * **Waveform selection via OTP forced temperature** (CCSET=0x02 + TSSET), exactly like
//!   Waveshare's `Init_Fast`/`Init_Part`, dexterdy and GxEPD2:
//!   `Full` = temperature-compensated OTP waveform, `Quick` = 0x5A, `PartialRefresh` = 0x6E.
//! * **Full / Quick never depend on controller RAM.** They write OLD = !NEW so that every
//!   pixel gets a transition waveform (what Waveshare's `Display()` effectively does).
//! * **PartialRefresh is differential.** It only drives pixels where OLD != NEW, so OLD must
//!   match what is physically on the glass. Within one power session N2OCP keeps it right.
//!   After the panel's deep sleep (0x07) or a power cut the SRAM is gone: call
//!   [`Epd7in5::restore_previous_partial_frame`] (window) or
//!   [`Epd7in5::restore_previous_frame`] (full screen) once after waking.
//! * **Window-limited refresh.** `update_partial_frame` only writes RAM inside the window and
//!   remembers it; `display_frame` then refreshes *only* that window
//!   (PartialIn → window → DRF → PartialOut). Garbage in RAM outside the window is never
//!   driven. (dexterdy's version writes the window but then runs a full-screen DRF, which is
//!   why everything outside the window was garbled after deep sleep.)
//!
//! Untested on hardware in this exact form — see the accompanying notes for the
//! bring-up order.

use embedded_hal::{
    delay::DelayNs,
    digital::{InputPin, OutputPin},
    spi::SpiDevice,
};
use embedded_hal_async::{delay::DelayNs as AsyncDelayNs, spi::SpiDevice as AsyncSpiDevice};

use crate::color::Color;
pub use crate::interface::BusyTimeoutError;
use crate::interface::DisplayInterface;
use crate::traits::{InternalWiAdditions, RefreshLut, WaveshareDisplay};

pub(crate) mod command;
use self::command::Command;
use crate::buffer_len;

/// Full size buffer for use with the 7in5 v2 EPD
#[cfg(feature = "graphics")]
pub type Display7in5 = crate::graphics::Display<
    WIDTH,
    HEIGHT,
    false,
    { buffer_len(WIDTH as usize, HEIGHT as usize) },
    Color,
>;

/// Width of the display
pub const WIDTH: u32 = 800;
/// Height of the display
pub const HEIGHT: u32 = 480;
/// Default Background Color
pub const DEFAULT_BACKGROUND_COLOR: Color = Color::White;
const IS_BUSY_LOW: bool = true;
const SINGLE_BYTE_WRITE: bool = false;

// ─── Register values ───────────────────────────────────────────────────────
//
// CDI (0x50) byte 1 = BDZ | BDV[1:0] | N2OCP | 0 | DDX[1:0]
// With DDX[0]=1 the {NEW,OLD} → LUT mapping is 11→WW, 10→KW, 01→WK, 00→KK,
// i.e. bit 1 = white in both planes.

/// BDV=10 (border uses LUTKW → white), N2OCP=1, DDX=01.
/// For a black border use 0x19 (BDV=01 → LUTWK).
const CDI_NORMAL: [u8; 2] = [0x29, 0x07];
/// As `CDI_NORMAL` but BDZ=1 (border floating) — Waveshare's partial-refresh value.
const CDI_PARTIAL: [u8; 2] = [0xA9, 0x07];
/// Waveshare `EPD_7IN5_V2_Sleep()` writes this before POWER OFF.
const CDI_SLEEP: [u8; 1] = [0xF7];

const POWER_SETTING: [u8; 4] = [0x07, 0x07, 0x3F, 0x3F];
const BOOSTER_NORMAL: [u8; 4] = [0x17, 0x17, 0x28, 0x17];
const BOOSTER_QUICK: [u8; 4] = [0x27, 0x27, 0x18, 0x17];
const PANEL_SETTING_KW_OTP: [u8; 1] = [0x1F];
const RESOLUTION: [u8; 4] = [0x03, 0x20, 0x01, 0xE0];
const DUAL_SPI_OFF: [u8; 1] = [0x00];
const TCON: [u8; 1] = [0x22];
const CCSET_SENSOR: [u8; 1] = [0x00];
const CCSET_TSFIX: [u8; 1] = [0x02];
/// Undocumented OTP waveform indices (Waveshare `Init_Fast` / `Init_Part`).
const TSSET_QUICK: [u8; 1] = [0x5A];
const TSSET_PARTIAL: [u8; 1] = [0x6E];
const DEEP_SLEEP_CHECK: [u8; 1] = [0xA5];

/// BUSY must be polled no earlier than ~200 µs after DRF (Waveshare comment).
const POST_REFRESH_DELAY_MS: u32 = 10;

// ─── Window ────────────────────────────────────────────────────────────────

/// A rectangle in native (unrotated) panel coordinates. `x` and `width` are multiples of 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Left edge, multiple of 8
    pub x: u32,
    /// Top edge
    pub y: u32,
    /// Width, multiple of 8
    pub width: u32,
    /// Height
    pub height: u32,
}

impl Window {
    /// Validates alignment and bounds.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        assert!(
            x % 8 == 0 && width % 8 == 0,
            "epd7in5_v2: x/width must be multiples of 8"
        );
        assert!(width > 0 && height > 0, "epd7in5_v2: empty window");
        assert!(
            x + width <= WIDTH && y + height <= HEIGHT,
            "epd7in5_v2: window out of bounds"
        );
        Window {
            x,
            y,
            width,
            height,
        }
    }

    /// Bytes needed for this window's buffer.
    pub const fn buffer_len(&self) -> usize {
        (self.width / 8 * self.height) as usize
    }

    fn union(self, o: Window) -> Window {
        let x0 = self.x.min(o.x);
        let y0 = self.y.min(o.y);
        let x1 = (self.x + self.width).max(o.x + o.width);
        let y1 = (self.y + self.height).max(o.y + o.height);
        Window {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        }
    }

    /// Payload for command 0x90 (inclusive end coordinates, PT_SCAN=1 like Waveshare).
    fn partial_window_bytes(&self) -> [u8; 9] {
        let xe = self.x + self.width - 1;
        let ye = self.y + self.height - 1;
        [
            (self.x >> 8) as u8,
            self.x as u8,
            (xe >> 8) as u8,
            xe as u8,
            (self.y >> 8) as u8,
            self.y as u8,
            (ye >> 8) as u8,
            ye as u8,
            0x01,
        ]
    }
}

/// Copies the bytes of `win` out of a full 800x480 framebuffer (native orientation)
/// into `out` (`win.buffer_len()` bytes). Useful to keep the "previous" window in RTC RAM.
pub fn extract_window(full: &[u8], win: Window, out: &mut [u8]) {
    let row = (WIDTH / 8) as usize;
    let wb = (win.width / 8) as usize;
    assert_eq!(full.len(), row * HEIGHT as usize);
    assert_eq!(out.len(), win.buffer_len());
    for r in 0..win.height as usize {
        let src = (win.y as usize + r) * row + (win.x / 8) as usize;
        out[r * wb..(r + 1) * wb].copy_from_slice(&full[src..src + wb]);
    }
}

// ─── Command sequences as data ─────────────────────────────────────────────
//
// Every operation is expressed once as a list of steps and executed by either the
// blocking or the async runner, so both paths are guaranteed to send identical bytes.

#[derive(Clone, Copy)]
enum Step<'a> {
    Nop,
    Reset,
    Cmd(Command),
    CmdData(Command, &'a [u8]),
    /// Command followed by bitwise-inverted data.
    CmdDataInverted(Command, &'a [u8]),
    /// Command followed by `count` copies of one byte.
    CmdFill(Command, u8, u32),
    DelayMs(u32),
    WaitIdle,
}

fn mode_steps(mode: RefreshLut) -> [Step<'static>; 4] {
    match mode {
        RefreshLut::Full => [
            Step::CmdData(Command::BoosterSoftStart, &BOOSTER_NORMAL),
            Step::CmdData(Command::VcomAndDataIntervalSetting, &CDI_NORMAL),
            Step::CmdData(Command::CascadeSetting, &CCSET_SENSOR),
            Step::Nop,
        ],
        RefreshLut::Quick => [
            Step::CmdData(Command::BoosterSoftStart, &BOOSTER_QUICK),
            Step::CmdData(Command::VcomAndDataIntervalSetting, &CDI_NORMAL),
            Step::CmdData(Command::CascadeSetting, &CCSET_TSFIX),
            Step::CmdData(Command::ForceTemperature, &TSSET_QUICK),
        ],
        RefreshLut::PartialRefresh => [
            Step::CmdData(Command::BoosterSoftStart, &BOOSTER_NORMAL),
            Step::CmdData(Command::VcomAndDataIntervalSetting, &CDI_PARTIAL),
            Step::CmdData(Command::CascadeSetting, &CCSET_TSFIX),
            Step::CmdData(Command::ForceTemperature, &TSSET_PARTIAL),
        ],
    }
}

/// Waveshare `EPD_7IN5_V2_Init` order (POWER ON before PSR/CDI and before the
/// waveform-selection registers), followed by the mode registers.
fn init_steps(mode: RefreshLut) -> [Step<'static>; 14] {
    let m = mode_steps(mode);
    [
        Step::Reset,
        Step::CmdData(Command::PowerSetting, &POWER_SETTING),
        Step::CmdData(Command::BoosterSoftStart, &BOOSTER_NORMAL),
        Step::Cmd(Command::PowerOn),
        Step::DelayMs(100),
        Step::WaitIdle,
        Step::CmdData(Command::PanelSetting, &PANEL_SETTING_KW_OTP),
        Step::CmdData(Command::TconResolution, &RESOLUTION),
        Step::CmdData(Command::DualSpi, &DUAL_SPI_OFF),
        Step::CmdData(Command::TconSetting, &TCON),
        // BTST, CDI, then CCSET + TSSET last, after POWER ON (as Waveshare and ESPHome do)
        m[0],
        m[1],
        m[2],
        m[3],
    ]
}

fn sleep_steps() -> [Step<'static>; 5] {
    [
        Step::WaitIdle,
        Step::CmdData(Command::VcomAndDataIntervalSetting, &CDI_SLEEP),
        Step::Cmd(Command::PowerOff),
        Step::WaitIdle,
        Step::CmdData(Command::DeepSleep, &DEEP_SLEEP_CHECK),
    ]
}

/// Writes NEW (and for Full/Quick also OLD = !NEW) inside a window.
fn write_window_steps<'a>(win: &'a [u8; 9], data: &'a [u8], force_all: bool) -> [Step<'a>; 5] {
    [
        Step::Cmd(Command::PartialIn),
        Step::CmdData(Command::PartialWindow, win),
        if force_all {
            Step::CmdDataInverted(Command::DataStartTransmission1, data)
        } else {
            Step::Nop
        },
        Step::CmdData(Command::DataStartTransmission2, data),
        Step::Cmd(Command::PartialOut),
    ]
}

/// Writes only OLD inside a window (deep-sleep recovery).
fn write_old_window_steps<'a>(win: &'a [u8; 9], data: &'a [u8]) -> [Step<'a>; 4] {
    [
        Step::Cmd(Command::PartialIn),
        Step::CmdData(Command::PartialWindow, win),
        Step::CmdData(Command::DataStartTransmission1, data),
        Step::Cmd(Command::PartialOut),
    ]
}

fn write_full_steps(data: &[u8], force_all: bool) -> [Step<'_>; 2] {
    [
        if force_all {
            Step::CmdDataInverted(Command::DataStartTransmission1, data)
        } else {
            Step::Nop
        },
        Step::CmdData(Command::DataStartTransmission2, data),
    ]
}

fn refresh_window_steps(win: &[u8; 9]) -> [Step<'_>; 6] {
    [
        Step::Cmd(Command::PartialIn),
        Step::CmdData(Command::PartialWindow, win),
        Step::Cmd(Command::DisplayRefresh),
        Step::DelayMs(POST_REFRESH_DELAY_MS),
        Step::WaitIdle,
        Step::Cmd(Command::PartialOut),
    ]
}

fn refresh_full_steps() -> [Step<'static>; 3] {
    [
        Step::Cmd(Command::DisplayRefresh),
        Step::DelayMs(POST_REFRESH_DELAY_MS),
        Step::WaitIdle,
    ]
}

fn clear_steps(color: Color) -> [Step<'static>; 2] {
    let n = WIDTH / 8 * HEIGHT;
    let new = color.get_byte_value();
    [
        Step::CmdFill(Command::DataStartTransmission1, !new, n),
        Step::CmdFill(Command::DataStartTransmission2, new, n),
    ]
}

// ─── Driver ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pending {
    Nothing,
    FullScreen,
    Window(Window),
}

/// Epd7in5 (V2) driver
pub struct Epd7in5<SPI, BUSY, DC, RST, DELAY> {
    interface: DisplayInterface<SPI, BUSY, DC, RST, DELAY, SINGLE_BYTE_WRITE>,
    color: Color,
    /// Waveform currently selected (re-applied on every wake-up).
    mode: RefreshLut,
    /// What `display_frame` has to refresh.
    pending: Pending,
}

// Bound-free helpers shared by the blocking and async impls.
impl<SPI, BUSY, DC, RST, DELAY> Epd7in5<SPI, BUSY, DC, RST, DELAY> {
    /// Currently selected waveform.
    pub fn refresh_mode(&self) -> RefreshLut {
        self.mode
    }

    /// Full/Quick drive every pixel (OLD = !NEW); PartialRefresh is differential.
    fn force_all(&self) -> bool {
        self.mode != RefreshLut::PartialRefresh
    }

    fn check_full(buffer: &[u8]) {
        assert_eq!(buffer.len(), buffer_len(WIDTH as usize, HEIGHT as usize));
    }

    fn check_window(buffer: &[u8], x: u32, y: u32, width: u32, height: u32) -> Window {
        let win = Window::new(x, y, width, height);
        assert_eq!(
            buffer.len(),
            win.buffer_len(),
            "epd7in5_v2: window buffer size"
        );
        win
    }

    fn note_full(&mut self) {
        self.pending = Pending::FullScreen;
    }

    fn note_window(&mut self, win: Window) {
        self.pending = match self.pending {
            Pending::Nothing => Pending::Window(win),
            Pending::Window(w) => Pending::Window(w.union(win)),
            Pending::FullScreen => Pending::FullScreen,
        };
    }
}

// ─── Blocking implementation ───────────────────────────────────────────────

impl<SPI, BUSY, DC, RST, DELAY> Epd7in5<SPI, BUSY, DC, RST, DELAY>
where
    SPI: SpiDevice,
    BUSY: InputPin,
    DC: OutputPin,
    RST: OutputPin,
    DELAY: DelayNs,
{
    fn run(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        steps: &[Step<'_>],
    ) -> Result<(), SPI::Error> {
        for step in steps {
            match *step {
                Step::Nop => {}
                Step::Reset => self.interface.reset(delay, 10_000, 2_000),
                Step::Cmd(c) => self.interface.cmd(spi, c)?,
                Step::CmdData(c, d) => self.interface.cmd_with_data(spi, c, d)?,
                Step::CmdDataInverted(c, d) => {
                    self.interface.cmd(spi, c)?;
                    self.interface.data_inverted(spi, d)?;
                }
                Step::CmdFill(c, v, n) => {
                    self.interface.cmd(spi, c)?;
                    let chunk = [v; 256];
                    let mut left = n as usize;
                    while left > 0 {
                        let k = left.min(chunk.len());
                        self.interface.data(spi, &chunk[..k])?;
                        left -= k;
                    }
                }
                Step::DelayMs(ms) => delay.delay_ms(ms),
                Step::WaitIdle => self.interface.wait_until_idle_with_cmd(
                    spi,
                    delay,
                    IS_BUSY_LOW,
                    Command::GetStatus,
                )?,
            }
        }
        Ok(())
    }

    /// Writes the image that is *currently on the glass* into OLD (DTM1), full screen.
    /// Needed after deep sleep before a full-screen `PartialRefresh`.
    pub fn restore_previous_frame(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        buffer: &[u8],
    ) -> Result<(), SPI::Error> {
        Self::check_full(buffer);
        self.run(
            spi,
            delay,
            &[Step::CmdData(Command::DataStartTransmission1, buffer)],
        )
    }

    /// Writes the window content that is *currently on the glass* into OLD (DTM1).
    /// Call once after waking the panel from deep sleep, before `update_partial_frame`.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_previous_partial_frame(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        buffer: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), SPI::Error> {
        let win = Self::check_window(buffer, x, y, width, height);
        let wb = win.partial_window_bytes();
        self.run(spi, delay, &write_old_window_steps(&wb, buffer))
    }
}

impl<SPI, BUSY, DC, RST, DELAY> InternalWiAdditions<SPI, BUSY, DC, RST, DELAY>
    for Epd7in5<SPI, BUSY, DC, RST, DELAY>
where
    SPI: SpiDevice,
    BUSY: InputPin,
    DC: OutputPin,
    RST: OutputPin,
    DELAY: DelayNs,
{
    fn init(&mut self, spi: &mut SPI, delay: &mut DELAY) -> Result<(), SPI::Error> {
        self.run(spi, delay, &init_steps(self.mode))?;
        self.pending = Pending::Nothing;
        Ok(())
    }
}

impl<SPI, BUSY, DC, RST, DELAY> WaveshareDisplay<SPI, BUSY, DC, RST, DELAY>
    for Epd7in5<SPI, BUSY, DC, RST, DELAY>
where
    SPI: SpiDevice,
    BUSY: InputPin,
    DC: OutputPin,
    RST: OutputPin,
    DELAY: DelayNs,
{
    type DisplayColor = Color;

    fn new(
        spi: &mut SPI,
        busy: BUSY,
        dc: DC,
        rst: RST,
        delay: &mut DELAY,
        delay_us: Option<u32>,
    ) -> Result<Self, SPI::Error> {
        let mut epd = Epd7in5 {
            interface: DisplayInterface::new(busy, dc, rst, delay_us),
            color: DEFAULT_BACKGROUND_COLOR,
            mode: RefreshLut::Full,
            pending: Pending::Nothing,
        };
        epd.init(spi, delay)?;
        Ok(epd)
    }

    /// Hardware reset + init, re-applying the current waveform mode
    /// (dexterdy's version kept `self.refresh` but lost the registers).
    fn wake_up(&mut self, spi: &mut SPI, delay: &mut DELAY) -> Result<(), SPI::Error> {
        self.init(spi, delay)
    }

    /// POWER OFF + DEEP SLEEP. Controller RAM is lost; registers are re-applied by `wake_up`.
    fn sleep(&mut self, spi: &mut SPI, delay: &mut DELAY) -> Result<(), SPI::Error> {
        self.run(spi, delay, &sleep_steps())?;
        self.pending = Pending::Nothing;
        Ok(())
    }

    fn update_frame(
        &mut self,
        spi: &mut SPI,
        buffer: &[u8],
        delay: &mut DELAY,
    ) -> Result<(), SPI::Error> {
        Self::check_full(buffer);
        let force = self.force_all();
        self.run(spi, delay, &write_full_steps(buffer, force))?;
        self.note_full();
        Ok(())
    }

    /// Writes `buffer` into the window. The refresh happens in `display_frame`, limited to the
    /// union of all windows written since the last refresh.
    fn update_partial_frame(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        buffer: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), SPI::Error> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        let win = Self::check_window(buffer, x, y, width, height);
        let wb = win.partial_window_bytes();
        let force = self.force_all();
        self.run(spi, delay, &write_window_steps(&wb, buffer, force))?;
        self.note_window(win);
        Ok(())
    }

    /// Refreshes what was written since the last refresh and waits for completion.
    fn display_frame(&mut self, spi: &mut SPI, delay: &mut DELAY) -> Result<(), SPI::Error> {
        match self.pending {
            Pending::Window(win) => {
                let wb = win.partial_window_bytes();
                self.run(spi, delay, &refresh_window_steps(&wb))?;
            }
            Pending::FullScreen | Pending::Nothing => {
                self.run(spi, delay, &refresh_full_steps())?;
            }
        }
        self.pending = Pending::Nothing;
        Ok(())
    }

    fn update_and_display_frame(
        &mut self,
        spi: &mut SPI,
        buffer: &[u8],
        delay: &mut DELAY,
    ) -> Result<(), SPI::Error> {
        self.update_frame(spi, buffer, delay)?;
        self.display_frame(spi, delay)
    }

    fn clear_frame(&mut self, spi: &mut SPI, delay: &mut DELAY) -> Result<(), SPI::Error> {
        let color = self.color;
        self.run(spi, delay, &clear_steps(color))?;
        self.note_full();
        self.display_frame(spi, delay)
    }

    fn set_background_color(&mut self, color: Color) {
        self.color = color;
    }

    fn background_color(&self) -> &Color {
        &self.color
    }

    fn width(&self) -> u32 {
        WIDTH
    }

    fn height(&self) -> u32 {
        HEIGHT
    }

    /// Switches waveform without a reset (controller RAM is untouched), as dexterdy does.
    fn set_lut(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        refresh_rate: Option<RefreshLut>,
    ) -> Result<(), SPI::Error> {
        self.mode = refresh_rate.unwrap_or(RefreshLut::Full);
        self.run(spi, delay, &mode_steps(self.mode))
    }

    fn wait_until_idle(&mut self, spi: &mut SPI, delay: &mut DELAY) -> Result<(), SPI::Error> {
        self.run(spi, delay, &[Step::WaitIdle])
    }
}

// ─── Async implementation (embassy) ────────────────────────────────────────

impl<SPI, BUSY, DC, RST, DELAY> Epd7in5<SPI, BUSY, DC, RST, DELAY>
where
    SPI: AsyncSpiDevice,
    BUSY: InputPin,
    DC: OutputPin,
    RST: OutputPin,
    DELAY: AsyncDelayNs,
{
    async fn run_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        steps: &[Step<'_>],
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        for step in steps {
            match *step {
                Step::Nop => {}
                Step::Reset => self.interface.reset_async(delay, 10_000, 2_000).await,
                Step::Cmd(c) => self.interface.cmd_async(spi, c).await?,
                Step::CmdData(c, d) => self.interface.cmd_with_data_async(spi, c, d).await?,
                Step::CmdDataInverted(c, d) => {
                    self.interface.cmd_async(spi, c).await?;
                    self.interface.data_inverted_async(spi, d).await?;
                }
                Step::CmdFill(c, v, n) => {
                    self.interface.cmd_async(spi, c).await?;
                    let chunk = [v; 256];
                    let mut left = n as usize;
                    while left > 0 {
                        let k = left.min(chunk.len());
                        self.interface.data_async(spi, &chunk[..k]).await?;
                        left -= k;
                    }
                }
                Step::DelayMs(ms) => delay.delay_ms(ms).await,
                Step::WaitIdle => {
                    let mut never = || false;
                    self.interface
                        .wait_until_idle_with_cmd_timeout_async(
                            spi,
                            delay,
                            IS_BUSY_LOW,
                            Command::GetStatus,
                            timeout_us,
                            &mut never,
                        )
                        .await?;
                }
            }
        }
        Ok(())
    }

    /// Async constructor: hardware reset + init in `mode`.
    #[allow(clippy::too_many_arguments)]
    pub async fn new_async(
        spi: &mut SPI,
        busy: BUSY,
        dc: DC,
        rst: RST,
        delay: &mut DELAY,
        delay_us: Option<u32>,
        mode: RefreshLut,
        timeout_us: u32,
    ) -> Result<Self, BusyTimeoutError<SPI::Error>> {
        let mut epd = Epd7in5 {
            interface: DisplayInterface::new(busy, dc, rst, delay_us),
            color: DEFAULT_BACKGROUND_COLOR,
            mode,
            pending: Pending::Nothing,
        };
        epd.wake_up_async(spi, delay, timeout_us).await?;
        Ok(epd)
    }

    /// Async `wake_up`.
    pub async fn wake_up_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        self.run_async(spi, delay, &init_steps(self.mode), timeout_us)
            .await?;
        self.pending = Pending::Nothing;
        Ok(())
    }

    /// Async `set_lut`.
    pub async fn set_lut_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        mode: RefreshLut,
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        self.mode = mode;
        self.run_async(spi, delay, &mode_steps(mode), timeout_us)
            .await
    }

    /// Async `restore_previous_partial_frame`.
    #[allow(clippy::too_many_arguments)]
    pub async fn restore_previous_partial_frame_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        buffer: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        let win = Self::check_window(buffer, x, y, width, height);
        let wb = win.partial_window_bytes();
        self.run_async(spi, delay, &write_old_window_steps(&wb, buffer), timeout_us)
            .await
    }

    /// Async `update_frame`.
    pub async fn update_frame_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        buffer: &[u8],
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        Self::check_full(buffer);
        let force = self.force_all();
        self.run_async(spi, delay, &write_full_steps(buffer, force), timeout_us)
            .await?;
        self.note_full();
        Ok(())
    }

    /// Async `update_partial_frame`.
    #[allow(clippy::too_many_arguments)]
    pub async fn update_partial_frame_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        buffer: &[u8],
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        let win = Self::check_window(buffer, x, y, width, height);
        let wb = win.partial_window_bytes();
        let force = self.force_all();
        self.run_async(
            spi,
            delay,
            &write_window_steps(&wb, buffer, force),
            timeout_us,
        )
        .await?;
        self.note_window(win);
        Ok(())
    }

    /// Async `display_frame` (waits for the refresh to finish).
    pub async fn display_frame_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        match self.pending {
            Pending::Window(win) => {
                let wb = win.partial_window_bytes();
                self.run_async(spi, delay, &refresh_window_steps(&wb), timeout_us)
                    .await?;
            }
            Pending::FullScreen | Pending::Nothing => {
                self.run_async(spi, delay, &refresh_full_steps(), timeout_us)
                    .await?;
            }
        }
        self.pending = Pending::Nothing;
        Ok(())
    }

    /// Async `sleep`.
    pub async fn sleep_async(
        &mut self,
        spi: &mut SPI,
        delay: &mut DELAY,
        timeout_us: u32,
    ) -> Result<(), BusyTimeoutError<SPI::Error>> {
        self.run_async(spi, delay, &sleep_steps(), timeout_us)
            .await?;
        self.pending = Pending::Nothing;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epd_size() {
        assert_eq!(WIDTH, 800);
        assert_eq!(HEIGHT, 480);
        assert_eq!(DEFAULT_BACKGROUND_COLOR, Color::White);
    }

    #[test]
    fn window_bytes_inclusive_end() {
        // Waveshare's C demo computes `x_end % 256 - 1` and breaks for x_end = 256/512/768.
        let w = Window::new(248, 0, 8, 256);
        assert_eq!(
            w.partial_window_bytes(),
            [0x00, 248, 0x00, 255, 0x00, 0x00, 0x00, 255, 0x01]
        );
        let w = Window::new(504, 10, 16, 1);
        assert_eq!(w.partial_window_bytes()[..4], [0x01, 0xF8, 0x02, 0x07]);
    }

    #[test]
    fn window_union() {
        let a = Window::new(0, 0, 16, 10);
        let b = Window::new(64, 20, 8, 5);
        assert_eq!(
            a.union(b),
            Window {
                x: 0,
                y: 0,
                width: 72,
                height: 25
            }
        );
    }

    #[test]
    fn extract() {
        let mut full = [0u8; (WIDTH / 8 * HEIGHT) as usize];
        full[(3 * WIDTH / 8 + 2) as usize] = 0xAB;
        let win = Window::new(16, 3, 8, 1);
        let mut out = [0u8; 1];
        extract_window(&full, win, &mut out);
        assert_eq!(out, [0xAB]);
    }
}
