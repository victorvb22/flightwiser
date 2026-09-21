import { Clock } from "lucide-react";

// Same amber-advisory chip language as ScoreAnomalie.tsx's
// out_of_training_scope warning -- an informational heads-up, not an error,
// so it borrows that established "amber, not red" vocabulary rather than
// introducing a new one for a single banner.
export function WakeupBanner({ visible }: { visible: boolean }) {
  return (
    // Collapses to nothing (not just hidden) when not visible, so it never
    // leaves a gap above the search form / history table -- same
    // maxHeight+overflow:hidden technique used for the expandable cards
    // elsewhere on these pages, just inline here since this is the only
    // place that needs it.
    <div style={{ maxHeight: visible ? 44 : 0, overflow: "hidden", transition: "max-height 500ms cubic-bezier(0.22, 1, 0.36, 1)" }}>
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
          transition: "opacity 350ms ease",
        }}
      >
        <Clock size={13} />
        The server is waking up from inactivity — this can take up to a minute.
      </p>
    </div>
  );
}
