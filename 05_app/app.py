"""
Application Streamlit : declaration de sinistre et detection de fraude.

Lancer en local :
    streamlit run 05_app/app.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# Permettre l'import depuis 03_src
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "03_src"))

st.set_page_config(
    page_title="Declaration de sinistre - Detection de fraude",
    page_icon="V",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cache : chargement du detecteur (une seule fois par session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement des modeles (10-30s la premiere fois)...")
def load_detector(variant: str, load_lmm: bool, load_judge: bool):
    from inference import FraudDetector
    return FraudDetector(
        model_dir=str(ROOT / "04_models"),
        variant=variant,
        load_lmm=load_lmm,
        load_judge=load_judge,
    )


# ---------------------------------------------------------------------------
# Sidebar : reglages techniques
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Reglages systeme")
    variant = st.radio(
        "Variante de modele",
        options=["lite", "full"],
        index=0,
        help="Lite : CLIP + texte handcrafted (rapide, deployable). Full : ajoute BLIP-2/LLaVA/Mistral (GPU recommande).",
    )
    threshold_low = st.slider("Seuil 'Legitime' (max)", 0.0, 1.0, 0.30, 0.05)
    threshold_high = st.slider("Seuil 'Fraude probable' (min)", 0.3, 1.0, 0.70, 0.05)
    if threshold_high < threshold_low:
        threshold_high = threshold_low + 0.05

    st.markdown("---")
    st.markdown("### Modeles avances (full only)")
    load_lmm = st.checkbox("Charger BLIP-2 + LLaVA", value=False, disabled=(variant == "lite"))
    load_judge = st.checkbox("Charger Mistral (LLM judge)", value=False, disabled=(variant == "lite"))
    st.caption("Necessite ~12 Go VRAM. Garder decoche sur HF Spaces gratuit.")


# ---------------------------------------------------------------------------
# En-tete
# ---------------------------------------------------------------------------

st.title("Declaration de sinistre habitation")
st.caption("Outil de detection de fraude par analyse d'image et de cohérence textuelle. Projet pédagogique - ISFA 2025-2026.")

# State pour passer du formulaire aux resultats
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
if "result" not in st.session_state:
    st.session_state["result"] = None
if "form_data" not in st.session_state:
    st.session_state["form_data"] = None
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = None


# ---------------------------------------------------------------------------
# FORMULAIRE
# ---------------------------------------------------------------------------

if not st.session_state["submitted"]:
    with st.form("claim_form", clear_on_submit=False):

        # Section 1
        st.markdown("### 1. Vos coordonnees")
        c1, c2, c3 = st.columns(3)
        with c1:
            last_name = st.text_input("Nom", value="DUPONT")
            email = st.text_input("Email", value="exemple@domaine.fr")
        with c2:
            first_name = st.text_input("Prenom", value="Jean")
            phone = st.text_input("Telephone", value="06 12 34 56 78")
        with c3:
            contract = st.text_input("N° de contrat", value="HAB-2024-00012345")

        # Section 2
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
            ownership = st.selectbox("Vous etes", ["Proprietaire occupant", "Locataire", "Proprietaire bailleur"])

        # Section 3
        st.markdown("### 3. Le sinistre")
        c1, c2 = st.columns(2)
        with c1:
            claim_type = st.selectbox(
                "Type de sinistre",
                options=[
                    ("water", "Degat des eaux"),
                    ("fire", "Incendie"),
                    ("glass", "Bris de vitre"),
                    ("vandalism", "Degradation / cambriolage"),
                ],
                format_func=lambda x: x[1],
            )[0]
            claim_date = st.date_input("Date de survenance", value=date.today() - timedelta(days=2))
            claim_time = st.time_input("Heure approximative")
        with c2:
            discovery_date = st.date_input("Date de decouverte", value=date.today() - timedelta(days=1))
            location = st.text_input("Piece concernee", value="Salon")
            present = st.radio("Etiez-vous present au moment du sinistre ?", ["Oui", "Non"], horizontal=True)

        circumstances = st.text_area(
            "Circonstances detaillees",
            value="",
            height=160,
            help="Decrivez precisement ce qui s'est passe : que faisiez-vous, qu'avez-vous constate, "
                 "y a-t-il eu un bruit, etc. 500 caracteres recommandes.",
            placeholder="Ex : Je suis rentre du travail vers 19h le 28 avril. En entrant dans le salon, j'ai immediatement constate une grande flaque d'eau au sol. Le plafond presentait une auréole humide...",
        )
        cause = st.text_input("Origine probable du sinistre", value="")

        # Section 4
        st.markdown("### 4. Estimation et objets endommages")
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Montant estime des dommages (EUR)",
                                     min_value=0, max_value=500000, value=2500, step=100)
        with c2:
            damaged_items = st.text_area(
                "Objets endommages (un par ligne)",
                value="",
                height=100,
                placeholder="Canape (achete 2022, ~1200 EUR)\nTele 55 pouces\nTapis salon",
            )

        # Section 5
        st.markdown("### 5. Tiers et temoins")
        c1, c2 = st.columns(2)
        with c1:
            third_party = st.checkbox("Tiers responsable identifie (voisin, artisan...)")
            third_party_info = st.text_input("Coordonnees du tiers", disabled=not third_party)
        with c2:
            witness = st.checkbox("Temoin present")
            witness_info = st.text_input("Coordonnees du temoin", disabled=not witness)

        # Section 6
        st.markdown("### 6. Mesures conservatoires et autorites")
        c1, c2 = st.columns(2)
        with c1:
            authorities = st.checkbox("Pompiers / police / gendarmerie sont intervenus")
            pv_number = st.text_input("Numero de PV ou main courante", disabled=not authorities)
        with c2:
            measures = st.text_area(
                "Mesures d'urgence prises",
                value="",
                placeholder="Ex : coupure eau, bachage du toit, contact syndic...",
                height=100,
            )

        # Section 7
        st.markdown("### 7. Photo du sinistre")
        uploaded = st.file_uploader(
            "Photo du sinistre (obligatoire)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            st.image(uploaded, caption="Apercu", width=400)

        # Soumission
        st.markdown("---")
        submit = st.form_submit_button("Soumettre la declaration", type="primary", use_container_width=True)

    if submit:
        if uploaded is None:
            st.error("Merci d'ajouter une photo du sinistre.")
            st.stop()
        if len(circumstances.strip()) < 30:
            st.warning("Description des circonstances un peu courte. Ajoutez des details si possible (>30 caracteres).")

        # Construction du dict form_data
        form_data = {
            "last_name": last_name, "first_name": first_name, "email": email, "phone": phone,
            "contract": contract,
            "address": address, "city": city, "postal": postal,
            "property_type": property_type, "surface": surface, "ownership": ownership,
            "claim_type": claim_type,
            "claim_date": claim_date.isoformat() if hasattr(claim_date, 'isoformat') else str(claim_date),
            "claim_time": claim_time.isoformat() if hasattr(claim_time, 'isoformat') else str(claim_time),
            "discovery_date": discovery_date.isoformat() if hasattr(discovery_date, 'isoformat') else str(discovery_date),
            "location": location, "present": present,
            "circumstances": circumstances, "cause": cause,
            "amount": amount, "damaged_items": damaged_items,
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


# ---------------------------------------------------------------------------
# RESULTATS
# ---------------------------------------------------------------------------

else:
    result = st.session_state["result"]
    form_data = st.session_state["form_data"]
    image_bytes = st.session_state["image_bytes"]

    # Recalcul de la decision avec les seuils choisis dans la sidebar
    score_used = result["score_used"]
    if score_used < threshold_low:
        decision = "Legitime"
        color = "#1b9e77"
    elif score_used < threshold_high:
        decision = "A expertiser (escalade humaine recommandee)"
        color = "#d95f02"
    else:
        decision = "Fraude probable"
        color = "#e7298a"

    st.markdown("## Resultat de l'analyse")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_bytes, caption="Image soumise", use_container_width=True)
    with col2:
        st.markdown(
            f"<div style='padding:1rem; background:{color}22; border-left:5px solid {color}; border-radius:6px;'>"
            f"<h3 style='color:{color}; margin:0;'>Decision : {decision}</h3>"
            f"<p style='margin:.5rem 0 0 0;'>Score utilise : <b>{score_used:.3f}</b></p>"
            f"</div>",
            unsafe_allow_html=True,
        )

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
        st.dataframe(
            {"Feature": list(feats.keys()), "Valeur": [round(v, 3) for v in feats.values()]},
            use_container_width=True, hide_index=True,
        )
        st.caption("EXIF=1 indique que la photo contient des metadonnees camera. Les scores LMM sont a 0.5 si non charges.")

    with tab2:
        text_feats = result.get("text_features", {})
        text_only = {k: v for k, v in text_feats.items() if k.startswith(("txt_", "judge_"))}
        if text_only:
            st.dataframe(
                {"Feature": list(text_only.keys()), "Valeur": [round(float(v), 3) for v in text_only.values()]},
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Aucune feature texte calculee.")

    with tab3:
        cross = result.get("text_features", {})
        cross_only = {k: v for k, v in cross.items() if k.startswith("cf_")}
        if cross_only:
            st.dataframe(
                {"Critere": list(cross_only.keys()), "Valeur": [round(float(v), 3) for v in cross_only.values()]},
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "cf_delay_days = ecart en jours entre survenance et decouverte. "
                "cf_amount_per_m2 = montant declare par m2. "
                "cf_circumstances_short = 1 si la description fait moins de 80 caracteres (signal d'alerte)."
            )
        else:
            st.info("Aucune coherence inter-champs calculee.")

    st.markdown("---")
    if st.button("Faire une nouvelle declaration", type="secondary"):
        st.session_state["submitted"] = False
        st.session_state["result"] = None
        st.session_state["form_data"] = None
        st.session_state["image_bytes"] = None
        st.rerun()
