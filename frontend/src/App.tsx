import { useEffect, useRef, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { RechercheVol } from "./pages/RechercheVol";
import { VueAgregee } from "./pages/VueAgregee";
import { Documentation } from "./pages/Documentation";

const navLinkStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
  textDecoration: "none",
  textTransform: "uppercase",
  letterSpacing: 1.5,
  fontFamily: "var(--font-mono)",
  fontSize: 13,
  fontWeight: isActive ? 400 : 300,
  color: isActive ? "var(--green)" : "rgba(255, 255, 255, 0.55)",
  transition: "color 120ms",
});

export default function App() {
  const headerRef = useRef<HTMLElement>(null);
  const navRef = useRef<HTMLElement>(null);
  // Where the fade reaches solid black — measured against "Documentation"'s
  // own right edge rather than a guessed percentage, so it tracks the
  // header regardless of viewport width or font metrics.
  const [fadePercent, setFadePercent] = useState(30);
  useEffect(() => {
    function measure() {
      const header = headerRef.current;
      const nav = navRef.current;
      if (!header || !nav) return;
      const headerBox = header.getBoundingClientRect();
      const navBox = nav.getBoundingClientRect();
      // A little past Documentation's own edge, not flush against it.
      const extraPx = 24;
      if (headerBox.width > 0) setFadePercent(((navBox.right - headerBox.left + extraPx) / headerBox.width) * 100);
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  return (
    <>
      <header
        ref={headerRef}
        className="radar-grid"
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          display: "flex",
          alignItems: "center",
          padding: "18px clamp(16px, 3vw, 48px)",
          background: "rgba(5, 6, 7, 0.82)",
          backdropFilter: "blur(10px)",
        }}
      >
        <nav ref={navRef} style={{ display: "flex", gap: 28 }}>
          <NavLink to="/" style={navLinkStyle} end>
            Search
          </NavLink>
          <NavLink to="/agregee" style={navLinkStyle}>
            Aggregate view
          </NavLink>
          <NavLink to="/documentation" style={navLinkStyle}>
            Documentation
          </NavLink>
        </nav>
        {/* Fades from green (50% opacity) down to solid black by the time
            it reaches Documentation's own right edge, then stays black the
            rest of the way — a deliberate accent under the nav rather than
            a plain border running the header's full width. */}
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: 0,
            height: 2,
            background: `linear-gradient(to right, rgba(47, 230, 164, 0.5) 0%, black ${fadePercent}%, black 100%)`,
          }}
        />
      </header>
      <main style={{ flex: 1, padding: "32px clamp(16px, 3vw, 48px)", width: "100%" }}>
        <Routes>
          <Route path="/" element={<RechercheVol />} />
          <Route path="/agregee" element={<VueAgregee />} />
          <Route path="/documentation" element={<Documentation />} />
        </Routes>
      </main>
    </>
  );
}
