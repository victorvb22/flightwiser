import type { Categorie } from "../services/api";

// Previously defined identically in both Documentation.tsx and
// VueAgregee.tsx (each with a comment pointing at the other one) -- a
// changed label needed remembering to edit both, with nothing to catch a
// missed one.
export const CATEGORY_LABELS: Record<Categorie, string> = {
  avion_ligne: "Airliner",
  jet_affaire: "Business jet",
  petit_avion: "Small aircraft",
  helicoptere: "Helicopter",
};
