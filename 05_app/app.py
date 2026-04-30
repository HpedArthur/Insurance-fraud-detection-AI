"""Application Streamlit : declaration de sinistre et detection de fraude."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "03_src"))

st.set_page_config(page_title="Declaration de sinistre - Detection de fraude",
                    page_icon="V", layout="wide")


@st.cache_resource(show_spinner="Chargement des modeles (10-30s la premiere fois)...")
def load_detector(variant, load_lmm, load_judge):
    from inference import FraudDetector
    return FraudDetector(model_dir=str(ROOT / "04_models"), variant=variant,
                         load_lmm=load_lmm, load_judge=load_judge)


# ============================================================
# SIDEBAR avec explications visibles
# ============================================================

with st.sidebar:
    st.markdown("## Reglages systeme")
    st.caption("Pour la premiere utilisation, garde les valeurs par defaut.")

    # --- Variante de modele ---
    with st.expander("Variante de modele : qu'est-ce que c'est ?", expanded=False):
        st.markdown(
            "**Lite** (defaut) : utilise CLIP + EXIF + features texte simples.\n"
            "Rapide (~5s par dossier), tourne sans GPU. C'est ce qui sera deploye.\n\n"
            "**Full** : ajoute BLIP-2, LLaVA et Mistral pour une analyse plus fine.\n"
            "Plus precis mais necessite une GPU avec ~12 Go de VRAM."
        )
    variant = st.radio("Variante", options=["lite", "full"], index=0, horizontal=True,
                       label_visibility="collapsed")

    st.markdown("---")

    # --- Seuils de decision ---
    st.markdown("### Seuils de decision")
    with st.expander("Comment fonctionnent les seuils ?", expanded=False):
        st.markdown(
            "Le modele produit un **score entre 0 et 1** (probabilite de fraude).\n\n"
            "Trois zones de decision :\n"
            "- **Score < seuil bas** : Legitime (auto-validation)\n"
            "- **Entre les deux seuils** : A expertiser (escalade humaine)\n"
            "- **Score > seuil haut** : Fraude probable\n\n"
            "**Important** : seuil bas doit etre **inferieur** au seuil haut.\n"
            "Reglages typiques : bas=0.30, haut=0.70 (equilibre)."
        )

    thresh_path = ROOT / "04_models" / f"thresholds_image_{variant}.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            cal = json.load(f)
        default_low = cal.get("low", 0.30)
        default_high = cal.get("high", 0.70)
        st.success(f"Seuils calibres automatiquement chargés\n"
                   f"(bas={default_low:.2f}, haut={default_high:.2f})")
    else:
        default_low, default_high = 0.30, 0.70
        st.info("Seuils par defaut. Lance `calibration.py` pour les optimiser sur votre dataset.")

    threshold_low = st.slider("Seuil bas (max pour Legitime)", 0.0, 1.0, default_low, 0.05,
                              help="En dessous de ce score, le dossier est considere legitime.")
    threshold_high = st.slider("Seuil haut (min pour Fraude)", 0.0, 1.0, default_high, 0.05,
                               help="Au dessus de ce score, le dossier est considere comme fraude probable.")
    if threshold_high < threshold_low:
        st.warning(f"Le seuil haut doit etre superieur au seuil bas. Ajuste automatiquement a {threshold_low + 0.05:.2f}.")
        threshold_high = threshold_low + 0.05

    st.markdown("---")

    # --- Heatmap ---
    with st.expander("Explication visuelle (heatmap) ?", expanded=False):
        st.markdown(
            "Calcule une **heatmap d'occlusion** sur l'image soumise.\n\n"
            "Les zones **rouges** sont celles que le modele a regardees pour decider.\n"
            "Tres utile pour justifier un verdict ou debugger.\n\n"
            "Cout : ajoute ~5 a 15 secondes par dossier. A decocher si trop lent."
        )
    show_explain = st.checkbox("Afficher la heatmap d'explication", value=True)

    st.markdown("---")

    # --- Modeles avances ---
    st.markdown("### Modeles avances")
    with st.expander("Quand activer ?", expanded=False):
        st.markdown(
            "Disponible uniquement en variante **full**.\n\n"
            "**BLIP-2 + LLaVA** : 2 LMM open-source qui analysent l'image et donnent\n"
            "des scores supplementaires (artefacts, coherence de scene).\n\n"
            "**Mistral judge** : LLM qui evalue le texte de declaration\n"
            "(coherence narrative, signaux d'alerte).\n\n"
            "Ces options consomment ~12 Go de VRAM cumules.\n"
            "**A garder decoche sur HuggingFace Spaces gratuit (CPU only).**"
        )
    load_lmm = st.checkbox("BLIP-2 + LLaVA (image)", value=False, disabled=(variant == "lite"))
    load_judge = st.checkbox("Mistral judge (texte)", value=False, disabled=(variant == "lite"))

    if variant == "lite":
        st.caption("Active la variante 'full' pour utiliser ces options.")

    st.markdown("---")
    st.caption("Astuce : les valeurs par defaut conviennent pour une premiere demo.")


# ============================================================
# CORPS PRINCIPAL
# ============================================================

st.title("Declaration de sinistre habitation")
st.caption("Outil de detection de fraude par analyse d'image et de coherence textuelle. "
           "Projet pedagogique - ISFA 2025-2026.")

for k, v in [("submitted", False), ("result", None), ("form_data", None), ("image_bytes", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


if not st.session_state["submitted"]:
    with st.form("claim_form", clear_on_submit=False):
        st.markdown("### 1. Vos coordonnees")
        c1, c2, c3 = st.columns(3)
        with c1:
            last_name = st.text_input("Nom", value="DUPONT")
            email = st.text_input("Email", value="exemple@domaine.fr")
        with c2:
            first_name = st.text_input("Prenom", value="Jean")
            phone = st.text_input("Telephone", value="06 12 34 56 78")
        with c3:
            contract = st.text_input("N de contrat", value="HAB-2024-00012345")

        st.markdown("### 2. Le bien sinistre")
        c1, c2, c3 = st.columns(3)
        with c1:
            address = st.text_input("Adresse", value="12 rue de la Republique")
            city = st.text_input("Ville", value="Lyon")
        with c2:
            postal = st.text_input("Code postal", value="69002")
            property_type = st.selectbox("Type de bien", ["Appartement", "Maison", "Studio", "Autre"])
        with c3:
            surface = st.number_input("Surface (m2)", min_value=10, max_value=1000, value=70)
            ownership = st.selectbox("Vous etes",
                ["Proprietaire occupant", "Locataire", "Proprietaire bailleur"])

        st.markdown("### 3. Le sinistre")
        c1, c2 = st.columns(2)
        with c1:
            claim_type = st.selectbox("Type de sinistre",
                options=[("water", "Degat des eaux"), ("fire", "Incendie"),
                         ("glass", "Bris de vitre"), ("vandalism", "Degradation / cambriolage")],
                format_func=lambda x: x[1])[0]
            claim_date = st.date_input("Date de survenance", value=date.today() - timedelta(days=2))
            claim_time = st.time_input("Heure approximative")
        with c2:
            discovery_date = st.date_input("Date de decouverte", value=date.today() - timedelta(days=1))
            location = st.text_input("Piece concernee", value="Salon")
            present = st.radio("Etiez-vous present au moment du sinistre ?", ["Oui", "Non"], horizontal=True)

        circumstances = st.text_area("Circonstances detaillees", value="", height=160,
            help="Decrivez precisement ce qui s'est passe. 500 caracteres recommandes.",
            placeholder="Ex : Je suis rentre du travail vers 19h. En entrant dans le salon, j'ai constate une flaque d'eau...")
        cause = st.text_input("Origine probable du sinistre", value="")

        st.markdown("### 4. Estimation et objets endommages")
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Montant estime des dommages (EUR)",
                                      min_value=0, max_value=500000, value=2500, step=100)
        with c2:
            damaged_items = st.text_area("Objets endommages (un par ligne)", value="", height=100,
                placeholder="Canape (achete 2022, ~1200 EUR)\nTele 55 pouces\nTapis salon")

        st.markdown("### 5. Tiers et temoins")
        c1, c2 = st.columns(2)
        with c1:
            third_party = st.checkbox("Tiers responsable identifie")
            third_party_info = st.text_input("Coordonnees du tiers", disabled=not third_party)
        with c2:
            witness = st.checkbox("Temoin present")
            witness_info = st.text_input("Coordonnees du temoin", disabled=not witness)

        st.markdown("### 6. Mesures conservatoires et autorites")
        c1, c2 = st.columns(2)
        with c1:
            authorities = st.checkbox("Pompiers / police / gendarmerie sont intervenus")
            pv_number = st.text_input("Numero de PV ou main courante", disabled=not authorities)
        with c2:
            measures = st.text_area("Mesures d'urgence prises", value="",
                placeholder="Ex : coupure eau, bachage du toit, contact syndic...", height=100)

        st.markdown("### 7. Photo du sinistre")
        uploaded = st.file_uploader("Photo du sinistre (obligatoire)",
                                      type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        if uploaded is not None:
            st.image(uploaded, caption="Apercu", width=400)

        st.markdown("---")
        submit = st.form_submit_button("Soumettre la declaration", type="primary",
                                        use_container_width=True)

    if submit:
        if uploaded is None:
            st.error("Merci d'ajouter une photo du sinistre."); st.stop()
        if len(circumstances.strip()) < 30:
            st.warning("Description un peu courte. Ajoutez des details si possible.")

        form_data = {
            "last_name": last_name, "first_name": first_name, "email": email, "phone": phone,
            "contract": contract, "address": address, "city": city, "postal": postal,
            "property_type": property_type, "surface": surface, "ownership": ownership,
            "claim_type": claim_type,
            "claim_date": claim_date.isoformat() if hasattr(claim_date, "isoformat") else str(claim_date),
            "claim_time": claim_time.isoformat() if hasattr(claim_time, "isoformat") else str(claim_time),
            "discovery_date": discovery_date.isoformat() if hasattr(discovery_date, "isoformat") else str(discovery_date),
            "location": location, "present": present, "circumstances": circumstances,
            "cause": cause, "amount": amount, "damaged_items": damaged_items,
            "third_party": third_party, "third_party_info": third_party_info,
            "witness": witness, "witness_info": witness_info,
            "authorities": authorities, "pv_number": pv_number, "measures": measures,
        }

        with st.spinner("Analyse en cours - extraction features image et texte, scoring..."):
            try:
                detector = load_detector(variant, load_lmm, load_judge)
                image_bytes = uploaded.read()
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                result = detector.predict(image, claim_form=form_data)
                st.session_state["result"] = result
                st.session_state["form_data"] = form_data
                st.session_state["image_bytes"] = image_bytes
                st.session_state["submitted"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")
                st.exception(e)

else:
    result = st.session_state["result"]
    form_data = st.session_state["form_data"]
    image_bytes = st.session_state["image_bytes"]

    score_used = result["score_used"]
    if score_used < threshold_low:
        decision, color = "Legitime", "#1b9e77"
    elif score_used < threshold_high:
        decision, color = "A expertiser (escalade humaine recommandee)", "#d95f02"
    else:
        decision, color = "Fraude probable", "#e7298a"

    st.markdown("## Resultat de l'analyse")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_bytes, caption="Image soumise", use_container_width=True)
        if show_explain:
            with st.spinner("Calcul de la heatmap d'explication..."):
                try:
                    detector = load_detector(variant, load_lmm, load_judge)
                    img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    img_small = img.copy(); img_small.thumbnail((384, 384))
                    explain = detector.explain_image(img_small, patch_size=48, stride=24)
                    st.image(explain["overlay"],
                              caption="Heatmap d'importance (rouge = zone utilisee par le modele)",
                              use_container_width=True)
                except Exception as e:
                    st.warning(f"Heatmap indisponible : {e}")
    with col2:
        st.markdown(
            f"<div style='padding:1rem; background:{color}22; border-left:5px solid {color}; border-radius:6px;'>"
            f"<h3 style='color:{color}; margin:0;'>Decision : {decision}</h3>"
            f"<p style='margin:.5rem 0 0 0;'>Score utilise : <b>{score_used:.3f}</b></p>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("### Decomposition par axe")
        st.metric("Score image (axe principal)", f"{result['score_image']:.3f}",
                  help="Probabilite que la photo soit generee par IA, basee sur l'image seule.")
        if result["score_multimodal"] is not None:
            delta = result["score_multimodal"] - result["score_image"]
            st.metric("Score multimodal (image + texte)", f"{result['score_multimodal']:.3f}",
                      delta=f"{delta:+.3f} vs image seule",
                      help="Score affine en integrant la coherence du texte de declaration.")
        else:
            st.info("Score multimodal indisponible (modele non charge).")

    st.markdown("---")
    st.markdown("### Detail des features")
    tab1, tab2, tab3 = st.tabs(["Image", "Texte", "Coherence formulaire"])

    with tab1:
        feats = result.get("image_features", {})
        st.write("Signaux extraits de la photo :")
        if feats:
            st.dataframe({"Feature": list(feats.keys()),
                          "Valeur": [round(float(v), 3) for v in feats.values()]},
                         use_container_width=True, hide_index=True)
        st.caption("EXIF=1 indique que la photo contient des metadonnees camera. "
                   "Les scores LMM sont a 0.5 si non charges.")

    with tab2:
        text_feats = result.get("text_features", {})
        text_only = {k: v for k, v in text_feats.items() if k.startswith(("txt_", "judge_"))}
        if text_only:
            st.dataframe({"Feature": list(text_only.keys()),
                          "Valeur": [round(float(v), 3) for v in text_only.values()]},
                         use_container_width=True, hide_index=True)
        else:
            st.info("Aucune feature texte calculee.")

    with tab3:
        cross = result.get("text_features", {})
        cross_only = {k: v for k, v in cross.items() if k.startswith("cf_")}
        if cross_only:
            st.dataframe({"Critere": list(cross_only.keys()),
                          "Valeur": [round(float(v), 3) for v in cross_only.values()]},
                         use_container_width=True, hide_index=True)
            st.caption("cf_delay_days = ecart en jours entre survenance et decouverte. "
                       "cf_amount_per_m2 = montant declare par m2. "
                       "cf_circumstances_short = 1 si la description fait moins de 80 caracteres.")
        else:
            st.info("Aucune coherence inter-champs calculee.")

    st.markdown("---")
    if st.button("Faire une nouvelle declaration", type="secondary"):
        for k in ["submitted", "result", "form_data", "image_bytes"]:
            st.session_state[k] = None if k != "submitted" else False
        st.rerun()
