import { Clock } from "lucide-react";

// Same amber-advisory chip language as ScoreAnomalie.tsx's
// out_of_training_scope warning -- an informational heads-up, not an error,
// so it borrows that established "amber, not red" vocabulary rather than
// introducing a new one for a single banner.
//
// max-height and opacity previously ran on different durations (500ms vs
// 350ms) -- disappearing, the text finished fading out 150ms before the
// now-invisible box was done collapsing, and appearing, the box was still
// growing for 150ms after the text had already reached full opacity: two
// motions finishing at different times rather than one coordinated reveal.
// A single shared duration/easing (cf. RechercheVol.tsx's own
// EXPAND_TRANSITION, same reasoning) makes the box and its content move
// together.
const WAKEUP_EASE = "420ms cubic-bezier(0.22, 1, 0.36, 1)";

export function WakeupBanner({ visible }: { visible: boolean }) {
  return (
    // Collapses to nothing (not just hidden) when not visible, so it never
    // leaves a gap above the search form / history table -- same
    // maxHeight+overflow:hidden technique used for the expandable cards
    // elsewhere on these pages, just inline here since this is the only
    // place that needs it. 56px, not the message's own ~34px natural
    // height, leaves headroom for the text to wrap to two lines on a
    // narrow phone without max-height itself clipping the second line.
    <div style={{ maxHeight: visible ? 56 : 0, overflow: "hidden", transition: `max-height ${WAKEUP_EASE}` }}>
      <p
        role="status"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: 12.5,
          color: "var(--amber)",
          background: "rgba(245, 185, 66, 0.12)",
          border: "1px solid rgba(245, 185, 66, 0.35)",
          borderRadius: 6,
          padding: "5px 9px",
          margin: "10px 0 0",
          opacity: visible ? 1 : 0,
          transition: `opacity ${WAKEUP_EASE}`,
        }}
      >
        <Clock size={13} />
        The server is waking up from inactivity — this can take up to a minute.
      </p>
    </div>
  );
}
