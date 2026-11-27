"""
Script to build and validate VISHWAS Knowledge Base JSONL
scripts/build_knowledge_base.py
"""

import json
from pathlib import Path

VISHWAS_ROOT = Path(__file__).resolve().parent.parent
KB_OUTPUT_PATH = VISHWAS_ROOT / "data" / "knowledge_base" / "kb_chunks.jsonl"

CHUNKS = [
    # =========================================================================
    # MEL: Melanoma
    # =========================================================================
    {
        "id": "kb_mel_typical_look",
        "class_code": "MEL",
        "section": "typical_look",
        "text": "Melanoma commonly presents as an asymmetric, irregularly bordered macule, patch, or plaque with variegated coloration spanning shades of brown, black, blue, red, and white. While many are larger than 6 mm, early lesions can be smaller. Nodular variants may present as rapidly enlarging firm, polypoid, or dome-shaped nodules that can be amelanotic.",
        "source": "Bolognia JL, Schaffer JV, Cerroni L. Dermatology, 4th Ed, Ch. 113 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_mel_dermoscopic_features",
        "class_code": "MEL",
        "section": "dermoscopic_features",
        "text": "Dermoscopic hallmarks include an atypical pigment network with thickened lines and irregular polygonal meshes, irregular dots and globules distributed non-uniformly, a blue-white veil (confluent structureless bluish ground-glass pigmentation), regression structures (peppering and scar-like depigmentation), atypical polymorphous vascular patterns (corkscrew, dotted, hairpin), and peripheral streaks or pseudopods.",
        "source": "Argenziano G et al. Seven-point checklist of dermoscopy: a new approach. Br J Dermatol 1998; 139:439-445",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_mel_differential",
        "class_code": "MEL",
        "section": "differential",
        "text": "Primary differentials include dysplastic melanocytic nevi, pigmented basal cell carcinoma, irritated seborrheic keratosis, thrombosed angioma, and pigmented dermatofibroma. The presence of milia-like cysts and crypts favors seborrheic keratosis, whereas arborizing telangiectasias and blue-gray ovoid nests favor basal cell carcinoma.",
        "source": "Soyer HP et al. Three-point checklist of dermoscopy. Dermatology 2004; 208(1):27-31",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_mel_risk_factors",
        "class_code": "MEL",
        "section": "risk_factors",
        "text": "Major risk factors include high intermittent UV radiation exposure, history of severe blistering sunburns in youth, personal or familial history of melanoma, presence of multiple dysplastic or congenital nevi (>50 common nevi), fair skin phototype (Fitzpatrick I-II), and germline CDKN2A or CDK4 mutations.",
        "source": "NCCN Clinical Practice Guidelines in Oncology: Melanoma, Version 2.2024",
        "tone_groups_relevant": ["G1", "G2"]
    },
    {
        "id": "kb_mel_red_flags",
        "class_code": "MEL",
        "section": "red_flags",
        "text": "Red flag symptoms include rapid evolution in size, shape, or elevation ('the ugly duckling' sign), spontaneous ulceration, friability, localized pain, pruritus, or new satellite pigmented papules surrounding an existing lesion. Any pigmented lesion exhibiting a parallel ridge pattern on palmar/plantar surfaces mandates urgent evaluation.",
        "source": "British Association of Dermatologists guidelines for the management of cutaneous melanoma. Br J Dermatol 2020; 183:801-819",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_mel_darker_skin_nuance",
        "class_code": "MEL",
        "section": "darker_skin_nuance",
        "text": "In darker skin types (Fitzpatrick IV-VI / Tone Group G3), melanoma disproportionately arises on non-sun-exposed acral sites including the soles of the feet, palms, and subungual nail beds (Acral Lentiginous Melanoma / ALM). Dermoscopically, ALM exhibits a diagnostic parallel ridge pattern (melanin bands along cutaneous dermatoglyphic ridges rather than furrows). Amelanotic or hypopigmented variants are frequently misdiagnosed as chronic ulcers, verrucae, or foreign body granulomas.",
        "source": "Bradford PT et al. Acral lentiginous melanoma: incidence and survival patterns in the United States. Arch Dermatol 2009; 145(4):427-434",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_mel_next_steps",
        "class_code": "MEL",
        "section": "next_steps",
        "text": "Suspected melanoma requires immediate urgent specialist referral to a dermatologist or surgical oncologist. The standard diagnostic procedure is a complete narrow-margin (1-3 mm) excisional biopsy down to the subcutaneous fat. Partial shave or punch biopsy should be avoided if total excision is clinically feasible to ensure accurate Breslow depth staging.",
        "source": "NCCN Guidelines for Melanoma: Cutaneous, Version 2.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # BCC: Basal Cell Carcinoma
    # =========================================================================
    {
        "id": "kb_bcc_typical_look",
        "class_code": "BCC",
        "section": "typical_look",
        "text": "Basal cell carcinoma typically manifests as a slow-growing, pearly translucent nodule with rolled borders and fine branching telangiectasia on sun-exposed head and neck regions. Central crusting, non-healing ulceration ('rodent ulcer'), and localized bleeding with minor trauma are frequent features. Superficial variants appear as well-circumscribed erythematous scaling plaques.",
        "source": "Bolognia JL, Schaffer JV, Cerroni L. Dermatology, 4th Ed, Ch. 108 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bcc_dermoscopic_features",
        "class_code": "BCC",
        "section": "dermoscopic_features",
        "text": "Key dermoscopic criteria include sharp, in-focus arborizing (tree-like) telangiectasias, blue-gray ovoid nests (discrete large round/oval pigment clumps), multiple blue-gray globules, spoke-wheel structures, leaf-like peripheral structures, shiny white blotches and short crossing strands visible under polarized light, and small micro-erosions.",
        "source": "Argenziano G et al. Dermoscopy of Basal Cell Carcinoma. Arch Dermatol 2002; 138(9):1157-1162",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bcc_differential",
        "class_code": "BCC",
        "section": "differential",
        "text": "Differential diagnoses include nodular melanoma, intradermal or compound melanocytic nevi, squamous cell carcinoma, actinic keratosis, and sebaceous hyperplasia. Sebaceous hyperplasia features central umbilication with crown vessels that curve around yellowish lobules rather than crossing over them.",
        "source": "Kittler H et al. Dermatoscopy: Pattern Analysis, 2nd Ed (Facultas, 2016)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bcc_risk_factors",
        "class_code": "BCC",
        "section": "risk_factors",
        "text": "Cumulative ultraviolet light exposure is the primary environmental risk factor. Additional predisposing elements include fair skin phototype, ionizing radiation therapy history, long-term immunosuppression (organ transplant recipients), and genetic syndromes such as Gorlin-Goltz (nevoid basal cell carcinoma syndrome).",
        "source": "BAD guidelines for the management of basal cell carcinoma. Br J Dermatol 2021; 185:899-915",
        "tone_groups_relevant": ["G1", "G2"]
    },
    {
        "id": "kb_bcc_red_flags",
        "class_code": "BCC",
        "section": "red_flags",
        "text": "Features indicating aggressive high-risk BCC subtypes (morpheaform, infiltrative, or micronodular) include indurated scar-like ill-defined borders, rapid lateral invasion, perineural pain or numbness, and proximity to high-risk 'H-zone' facial structures (eyelids, nose, ears, lips).",
        "source": "NCCN Guidelines: Basal Cell Skin Cancer, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bcc_darker_skin_nuance",
        "class_code": "BCC",
        "section": "darker_skin_nuance",
        "text": "In darker skin populations (Tone Group G3), over 50% of basal cell carcinomas are heavily pigmented (compared to under 10% in light-skinned patients). Pigmented BCC can present as dark brown, slate-gray, or jet-black nodular lesions that clinically mimic nodular melanoma or seborrheic keratosis. Dermoscopic identification of blue-gray ovoid nests and short shiny white strands provides critical differentiation.",
        "source": "Bigby M et al. Pigmented basal cell carcinoma in dark-skinned individuals. JAAD 2004; 51(5):789-794",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_bcc_next_steps",
        "class_code": "BCC",
        "section": "next_steps",
        "text": "Clinical evaluation by a dermatologist with dermoscopy is indicated. Standard treatment modalities include Mohs micrographic surgery (for high-risk facial H-zone or recurrent lesions), surgical excision with 4-5 mm margins, or electrodessication and curettage for low-risk superficial trunk lesions.",
        "source": "NCCN Guidelines: Basal Cell Skin Cancer, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # AKIEC: Actinic Keratosis / Bowen's Disease
    # =========================================================================
    {
        "id": "kb_akiec_typical_look",
        "class_code": "AKIEC",
        "section": "typical_look",
        "text": "Actinic keratosis presents as a rough, keratotic sandpaper-like macule or thin plaque on chronically sun-damaged skin. Color varies from pinkish-red to yellowish-brown. Bowen's disease (squamous cell carcinoma in situ) typically presents as a sharply circumscribed, slowly enlarging erythematous plaque with crusting or scaling.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 110 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_akiec_dermoscopic_features",
        "class_code": "AKIEC",
        "section": "dermoscopic_features",
        "text": "Dermoscopy of actinic keratosis demonstrates the classic 'strawberry pattern', consisting of an erythematous pseudo-network surrounding prominent follicular openings filled with yellowish-white keratotic plugs. Under polarized dermoscopy, fine rosettes (four white dots arranged as a clover) may be seen. Bowen's disease shows clustered glomerular (coiled) vessels and prominent surface hyperkeratosis.",
        "source": "Soyer HP et al. Dermoscopy of Actinic Keratosis and Bowen's Disease. JDDG 2015; 13:304-311",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_akiec_differential",
        "class_code": "AKIEC",
        "section": "differential",
        "text": "Differential diagnosis includes superficial basal cell carcinoma, seborrheic keratosis, discoid lupus erythematosus, and invasive squamous cell carcinoma. The emergence of dermal induration, tenderness, or rapid thickening signals possible progression to invasive SCC.",
        "source": "Kittler H et al. Dermatoscopy: Pattern Analysis, 2nd Ed (Facultas, 2016)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_akiec_risk_factors",
        "class_code": "AKIEC",
        "section": "risk_factors",
        "text": "Chronic cumulative UV exposure, older age, male sex, fair complexion (Fitzpatrick I-II), outdoor occupations, and pharmacological immunosuppression are prominent risk factors. Organ transplant recipients experience a significantly higher incidence of multiplicity and aggressive progression.",
        "source": "NCCN Guidelines: Squamous Cell Skin Cancer, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2"]
    },
    {
        "id": "kb_akiec_red_flags",
        "class_code": "AKIEC",
        "section": "red_flags",
        "text": "Red flags signaling progression from in situ actinic keratosis to invasive squamous cell carcinoma include localized pain, induration upon palpation, rapid elevation or nodularity, bleeding, and ulceration exceeding 1 cm.",
        "source": "BAD guidelines for the management of actinic keratoses. Br J Dermatol 2017; 176:868-888",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_akiec_darker_skin_nuance",
        "class_code": "AKIEC",
        "section": "darker_skin_nuance",
        "text": "Actinic keratosis is relatively uncommon in Fitzpatrick IV-VI skin due to melanin photoprotection. When present, baseline erythema is masked, causing lesions to appear as hyperpigmented brownish-gray macules or rough keratotic papules that can be misidentified as solar lentigines or seborrheic keratoses. Pigmented Bowen's disease must be differentiated from lentigo maligna melanoma.",
        "source": "Dadzie OE et al. Ethnic Dermatology: Principles and Practice, Ch. 14 (Wiley, 2013)",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_akiec_next_steps",
        "class_code": "AKIEC",
        "section": "next_steps",
        "text": "Treatment options depend on lesion count and field cancerization: cryotherapy for discrete isolated lesions; field-directed therapies (topical 5-fluorouracil, imiquimod, or photodynamic therapy) for widespread actinic damage. Biopsy is indicated if induration or rapid growth is noted.",
        "source": "BAD guidelines for the management of actinic keratoses. Br J Dermatol 2017",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # BKL: Benign Keratosis (Seborrheic Keratosis / Solar Lentigo)
    # =========================================================================
    {
        "id": "kb_bkl_typical_look",
        "class_code": "BKL",
        "section": "typical_look",
        "text": "Benign keratoses encompass seborrheic keratoses, solar lentigines, and lichen planus-like keratoses (LPLK). Seborrheic keratoses typically appear as sharply demarcated 'stuck-on' waxy, verrucous, or greasy papules or plaques ranging from light tan to deep black. Solar lentigines present as smooth, uniform light-brown macules on chronic sun-exposed skin.",
        "source": "Braun RP et al. Dermoscopy of seborrheic keratosis. JAMA Dermatol 2002; 138(4):518-523",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bkl_dermoscopic_features",
        "class_code": "BKL",
        "section": "dermoscopic_features",
        "text": "Pathognomonic dermoscopic features include milia-like cysts (small bright white/yellowish globules), comedo-like openings / crypts (sharp brown-black craters), gyri and sulci forming a brain-like (cerebriform) architecture, hairpin vessels surrounded by a white halo, and delicate fingerprint-like regular parallel ridges.",
        "source": "Braun RP et al. JAMA Dermatol 2002; 138:518-523",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bkl_differential",
        "class_code": "BKL",
        "section": "differential",
        "text": "Differential diagnoses include melanoma, pigmented basal cell carcinoma, melanocytic nevi, and verruca vulgaris. An irritated or inflamed seborrheic keratosis can mimic melanoma clinically, but preservation of comedo-like openings and absence of atypical network or blue-white veil assist differentiation.",
        "source": "Kittler H et al. Dermatoscopy, 2nd Ed (Facultas, 2016)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bkl_risk_factors",
        "class_code": "BKL",
        "section": "risk_factors",
        "text": "Advancing chronological age and cumulative lifetime sun exposure are major predisposing factors. Genetic factors and somatic mutations in FGFR3 and PIK3CA contribute to clonal seborrheic keratosis development.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 107 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bkl_red_flags",
        "class_code": "BKL",
        "section": "red_flags",
        "text": "While seborrheic keratoses are strictly benign, sudden explosive eruptive onset of hundreds of pruritic seborrheic keratoses (the sign of Leser-Trélat) can represent an obligate cutaneous paraneoplastic syndrome associated with underlying internal gastrointestinal adenocarcinomas.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 107",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_bkl_darker_skin_nuance",
        "class_code": "BKL",
        "section": "darker_skin_nuance",
        "text": "In darker skin types (Fitzpatrick IV-VI), seborrheic keratosis frequently manifests as Dermatosis Papulosa Nigra (DPN)—multiple 1-5 mm hyperpigmented, smooth, sessile or pedunculated papules distributed bilaterally over the malar cheeks and neck. Clinically benign, these lesions show cerebriform or crypt-like patterns under dermoscopy.",
        "source": "Babapour R et al. Dermatosis papulosa nigra: A review. JAAD 2017; 77(4):755-763",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_bkl_next_steps",
        "class_code": "BKL",
        "section": "next_steps",
        "text": "Benign keratoses require no therapeutic intervention unless symptomatic (pruritus, friction irritation) or cosmetically concerning. Conservative treatment options include cryotherapy, curettage, electrodessication, or light shave excision. Biopsy is indicated if dermoscopic criteria for malignancy cannot be definitively excluded.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # NV: Melanocytic Nevus (Common Mole)
    # =========================================================================
    {
        "id": "kb_nv_typical_look",
        "class_code": "NV",
        "section": "typical_look",
        "text": "Melanocytic nevi are benign neoplasms of melanocytes presenting as symmetric, uniformly pigmented macules, papules, or soft domes with sharp, regular borders. Coloration is typically homogeneous brown, tan, or flesh-toned, and lesion diameter is usually less than 6 mm.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 112 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_nv_dermoscopic_features",
        "class_code": "NV",
        "section": "dermoscopic_features",
        "text": "Characteristic dermoscopic patterns include a delicate regular pigment network that thins and fades gradually at the perimeter, uniform central or concentric peripheral brown globules, homogeneous light brown structureless pigmentation, or regular comma vessels in intradermal nevi. Malignant criteria are absent.",
        "source": "Kittler H et al. Dermatoscopy: Pattern Analysis, 2nd Ed (Facultas, 2016)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_nv_differential",
        "class_code": "NV",
        "section": "differential",
        "text": "Differential diagnosis includes melanoma, dysplastic nevus, seborrheic keratosis, dermatofibroma, and pigmented basal cell carcinoma. In contrast to melanoma, common nevi exhibit bilateral symmetry in color and architecture, with uniform peripheral transition to normal skin.",
        "source": "Argenziano G et al. Br J Dermatol 1998; 139:439-445",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_nv_risk_factors",
        "class_code": "NV",
        "section": "risk_factors",
        "text": "Nevus counts correlate with genetic predisposition, fair skin phototypes, childhood sun exposure, and episodes of intermittent blistering sunburns. The presence of large numbers of atypical or dysplastic nevi represents a phenotypic marker of elevated lifetime melanoma risk.",
        "source": "NCCN Guidelines: Melanoma, Version 2.2024",
        "tone_groups_relevant": ["G1", "G2"]
    },
    {
        "id": "kb_nv_red_flags",
        "class_code": "NV",
        "section": "red_flags",
        "text": "Any mole demonstrating the ABCDE criteria (Asymmetry, Border irregularity, Color variegation, Diameter >6 mm, or Evolution in size/shape/color) or violating the patient's individual nevus signature pattern ('ugly duckling') warrants prompt clinical assessment.",
        "source": "British Association of Dermatologists guidelines for cutaneous melanoma. Br J Dermatol 2020",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_nv_darker_skin_nuance",
        "class_code": "NV",
        "section": "darker_skin_nuance",
        "text": "In darker skin types (Fitzpatrick IV-VI / Tone Group G3), melanocytic nevi typically contain denser melanin pigmentation, frequently appearing dark brown or slate-black with high contrast. Reticular and homogeneous dark patterns predominate. Dermatologists must avoid misinterpreting high melanin concentration as malignant atypia when architectural symmetry is maintained.",
        "source": "Marghoob AA et al. Dermoscopic evaluation of nevi across diverse skin types. Semin Cutan Med Surg 2009; 28(3):165-171",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_nv_next_steps",
        "class_code": "NV",
        "section": "next_steps",
        "text": "Stable benign nevi require no treatment. Patients should be counseled on sun protection and monthly skin self-examinations. For suspicious or evolving lesions, complete excisional biopsy with 1-2 mm margins is indicated for formal histopathological evaluation.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # DF: Dermatofibroma
    # =========================================================================
    {
        "id": "kb_df_typical_look",
        "class_code": "DF",
        "section": "typical_look",
        "text": "Dermatofibroma is a benign fibrohistiocytic lesion presenting as a firm, solitary dermal nodule measuring 3 to 10 mm, predominantly located on the lower extremities of young to middle-aged adults. It is characteristically attached to the overlying skin and produces a positive 'dimple sign' (central depression) when pinched laterally.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 116 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_df_dermoscopic_features",
        "class_code": "DF",
        "section": "dermoscopic_features",
        "text": "The classic dermoscopic hallmark of dermatofibroma is a central white patch or structureless scar-like area surrounded by a delicate, uniform peripheral pigment network. Less common patterns include completely homogeneous pigmentation, delicate network throughout, or central shiny white crystalline lines.",
        "source": "Zaballos P et al. Dermoscopic patterns of dermatofibroma: a study of 412 cases. Arch Dermatol 2008; 144(1):55-60",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_df_differential",
        "class_code": "DF",
        "section": "differential",
        "text": "Differential diagnosis includes malignant melanoma, dermatofibrosarcoma protuberans (DFSP), basal cell carcinoma, and blue nevus. In DFSP, lesions are typically larger, multinodular, infiltrative, and lack the central scar-like white patch with delicate peripheral network.",
        "source": "Zaballos P et al. Arch Dermatol 2008",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_df_risk_factors",
        "class_code": "DF",
        "section": "risk_factors",
        "text": "Dermatofibromas frequently arise as an exaggerated reactive inflammatory or reparative response following minor localized trauma, such as arthropod bites, folliculitis, or superficial puncture injuries.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_df_red_flags",
        "class_code": "DF",
        "section": "red_flags",
        "text": "Rapid progressive expansion (>15 mm), subcutaneous fixation without surface attachment, ulceration, or multiple eruptive lesions (which can occur in systemic lupus erythematosus or advanced immunosuppression) require formal histopathological assessment to rule out sarcomatous pathology.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 116",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_df_darker_skin_nuance",
        "class_code": "DF",
        "section": "darker_skin_nuance",
        "text": "In patients with darker skin tones (Fitzpatrick IV-VI / Tone Group G3), the central scar-like region of a dermatofibroma often exhibits pronounced post-inflammatory hyperpigmentation rather than a pale porcelain-white patch. The peripheral network is frequently hypermelanotic, which can lead to clinical suspicion of melanoma.",
        "source": "Zaballos P et al. Arch Dermatol 2008; 144:55-60",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_df_next_steps",
        "class_code": "DF",
        "section": "next_steps",
        "text": "Reassurance is the primary management strategy for confirmed dermatofibroma. Routine surgical excision is generally discouraged because it creates an elongated scar that may be more noticeable than the original lesion. If excision is pursued, complete removal to the deep subcutaneous fat is necessary to avoid local recurrence.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # VASC: Vascular Lesion
    # =========================================================================
    {
        "id": "kb_vasc_typical_look",
        "class_code": "VASC",
        "section": "typical_look",
        "text": "Vascular lesions comprise cherry hemangiomas, pyogenic granulomas, venous lakes, and angiokeratomas. Cherry hemangiomas appear as small, bright ruby-red to violaceous well-circumscribed papules. Pyogenic granulomas present as rapidly growing, friable vascular papules or polypoid nodules that bleed profusely with minimal trauma.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 106 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_vasc_dermoscopic_features",
        "class_code": "VASC",
        "section": "dermoscopic_features",
        "text": "The diagnostic dermoscopic hallmark of vascular lesions is the presence of red, purple, or blue-black round/oval lacunae (vascular lagoons) representing dilated ectatic dermal blood spaces. These are frequently separated by pale whitish-pink fibrous septa. Homogeneous red-purple structureless areas are common; melanin network is completely absent.",
        "source": "Zaballos P et al. Dermoscopy of vascular lesions: a study of 422 cases. Arch Dermatol 2007; 143(3):329-335",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_vasc_differential",
        "class_code": "VASC",
        "section": "differential",
        "text": "Differential diagnosis includes nodular melanoma, amelanotic melanoma, Kaposi sarcoma, and pigmented basal cell carcinoma. Thrombosed hemangiomas can turn dark blue-black and mimic melanoma clinically, but residual non-thrombosed lacunae and lack of pigment network confirm a vascular nature.",
        "source": "Zaballos P et al. Arch Dermatol 2007",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_vasc_risk_factors",
        "class_code": "VASC",
        "section": "risk_factors",
        "text": "Cherry angiomas correlate strongly with increasing age and genetic predisposition. Pyogenic granulomas frequently follow micro-trauma, localized foreign body reactions, or hormonal alterations during pregnancy.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_vasc_red_flags",
        "class_code": "VASC",
        "section": "red_flags",
        "text": "Rapidly ulcerating, persistently bleeding, or non-healing polypoid vascular lesions must be carefully evaluated to rule out amelanotic melanoma, Kaposi sarcoma, or angiosarcoma. Lack of blanching under pressure (diascopy) in an atypical lesion warrants biopsy.",
        "source": "NCCN Guidelines: Melanoma, Version 2.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_vasc_darker_skin_nuance",
        "class_code": "VASC",
        "section": "darker_skin_nuance",
        "text": "In darker skin tones (Tone Group G3), vascular lacunae frequently appear dark violaceous, dark blue, or deep black rather than bright cherry red, because overlying epidermal melanin absorbs red light. Angiokeratomas in darker skin phototypes can closely simulate nodular melanoma.",
        "source": "Zaballos P et al. Arch Dermatol 2007; 143:329-335",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_vasc_next_steps",
        "class_code": "VASC",
        "section": "next_steps",
        "text": "Benign cherry hemangiomas require no treatment. If bleeding or symptomatic, electrodessication, pulsed dye laser (PDL), or curettage provides definitive management. Pyogenic granulomas require shave or surgical excision with base cauterization and histological verification.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # SCCKA: Squamous Cell Carcinoma / Keratoacanthoma
    # =========================================================================
    {
        "id": "kb_sccka_typical_look",
        "class_code": "SCCKA",
        "section": "typical_look",
        "text": "Cutaneous squamous cell carcinoma presents as an indurated, firm, hyperkeratotic nodule or plaque, often with central ulceration or a keratin crust, situated on sun-exposed skin. Keratoacanthoma (a clinical variant of SCC) exhibits a characteristic volcano-like dome shape with a prominent central crateriform keratinaceous core, developing rapidly over 4 to 8 weeks.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 109 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_sccka_dermoscopic_features",
        "class_code": "SCCKA",
        "section": "dermoscopic_features",
        "text": "Dermoscopic findings of SCC and keratoacanthoma include a central amorphous keratin mass, surface crusting or ulceration, white circles surrounding enlarged follicular openings, hairpin vessels surrounded by a distinctive white halo, and linear-irregular or serpentine atypical vessels at the lesion periphery.",
        "source": "Lallas A et al. Dermoscopy of Squamous Cell Carcinoma and Keratoacanthoma. Dermatol Pract Concept 2013; 3(2):61-68",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_sccka_differential",
        "class_code": "SCCKA",
        "section": "differential",
        "text": "Differential diagnoses include hypertrophic actinic keratosis, verrucous seborrheic keratosis, verruca vulgaris, and ulcerated basal cell carcinoma. Prominent white circles and central keratin with hairpin vessels strongly favor SCC/KA over BCC.",
        "source": "Lallas A et al. Dermatol Pract Concept 2013",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_sccka_risk_factors",
        "class_code": "SCCKA",
        "section": "risk_factors",
        "text": "Chronic cumulative UV exposure, older age, light skin phototype, and chronic human papillomavirus (HPV) infection contribute to risk. Solid organ transplant recipients under lifelong immunosuppression experience an extraordinary 65- to 250-fold increased risk of developing cutaneous SCC.",
        "source": "NCCN Guidelines: Squamous Cell Skin Cancer, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2"]
    },
    {
        "id": "kb_sccka_red_flags",
        "class_code": "SCCKA",
        "section": "red_flags",
        "text": "High-risk features for metastasis include tumor diameter >2 cm, depth of invasion >4 mm, perineural invasion, poorly differentiated histology, and high-risk anatomical locations (ear, lip, temple, genitalia).",
        "source": "NCCN Guidelines: Squamous Cell Skin Cancer, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_sccka_darker_skin_nuance",
        "class_code": "SCCKA",
        "section": "darker_skin_nuance",
        "text": "In darker-skinned populations (Fitzpatrick IV-VI / Tone Group G3), cutaneous squamous cell carcinoma is the single most common skin malignancy. Unlike in fair-skinned patients where it is predominantly UV-induced, SCC in darker skin frequently develops in non-sun-exposed areas, chronic burn scars (Marjolin ulcer), chronic venous stasis ulcers, or areas of chronic inflammation. They are frequently pigmented and exhibit higher rates of metastasis due to delayed clinical recognition.",
        "source": "Gloster HM, Neal K. Skin cancer in skin of color. J Am Acad Dermatol 2006; 55(5):741-760",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_sccka_next_steps",
        "class_code": "SCCKA",
        "section": "next_steps",
        "text": "Urgent dermatological assessment and histological diagnosis via biopsy are indicated. Standard treatment is surgical excision with 4 to 6 mm margins for low-risk tumors, or Mohs micrographic surgery for high-risk, cosmetically sensitive, or recurrent lesions.",
        "source": "NCCN Guidelines: Squamous Cell Skin Cancer, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # INF: Inflammatory Condition
    # =========================================================================
    {
        "id": "kb_inf_typical_look",
        "class_code": "INF",
        "section": "typical_look",
        "text": "Inflammatory dermatoses encompass eczema, psoriasis, and lichen planus. Eczematous lesions present as pruritic, ill-defined erythematous macules, papules, or plaques with scaling and excoriations. Plaque psoriasis presents as sharply circumscribed erythematous plaques covered by coarse silvery-white micaceous scales.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 8 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_inf_dermoscopic_features",
        "class_code": "INF",
        "section": "dermoscopic_features",
        "text": "Dermoscopy of inflammatory skin conditions (inflammoscopy) displays distinctive vessel architectures: psoriasis characteristically demonstrates uniformly distributed dotted vessels on a diffuse light-red background with silvery-white scales. Eczema displays patchy, non-uniform dotted vessels with yellow-orange serocrusts. Lichen planus demonstrates white crossing striae (Wickham striae).",
        "source": "Errichetti E, Stinco G. Dermoscopy in General Dermatology: A Practical Overview. Dermatol Ther 2016; 6(4):471-507",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_inf_differential",
        "class_code": "INF",
        "section": "differential",
        "text": "Differential diagnosis includes cutaneous T-cell lymphoma (mycosis fungoides), superficial basal cell carcinoma, Bowen's disease, and tinea incognito. Solitary, treatment-resistant plaques should undergo skin biopsy to exclude cutaneous neoplasia.",
        "source": "Errichetti E, Stinco G. Dermatol Ther 2016",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_inf_risk_factors",
        "class_code": "INF",
        "section": "risk_factors",
        "text": "Etiology involves genetic atopic diathesis, immune dysregulation (Th1/Th17 pathway activation in psoriasis; Th2 activation in atopic dermatitis), environmental barrier disruption, stress, and cutaneous micro-biome alterations.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_inf_red_flags",
        "class_code": "INF",
        "section": "red_flags",
        "text": "Generalized erythroderma (>90% body surface area involvement), intractable shivering, secondary bacterial infection (impetiginization or cellulitis), or persistent unresponsiveness to high-potency topical corticosteroids warrants comprehensive specialist dermatology evaluation.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_inf_darker_skin_nuance",
        "class_code": "INF",
        "section": "darker_skin_nuance",
        "text": "In darker skin phototypes (Fitzpatrick IV-VI / Tone Group G3), erythema rarely presents as classic bright pink or red. It frequently appears as violaceous, grayish-brown, or hyperpigmented plaques. Post-inflammatory hyperpigmentation or hypopigmentation is often severe, long-lasting, and clinically more distressing to the patient than the primary inflammatory rash itself.",
        "source": "Alexis AF et al. Dermatology for Skin of Color, 2nd Ed, McGraw-Hill, 2014",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_inf_next_steps",
        "class_code": "INF",
        "section": "next_steps",
        "text": "Management comprises skin barrier restoration (emollients, soap substitutes), topical anti-inflammatory agents (corticosteroids, calcineurin inhibitors, PDE4 inhibitors), and phototherapy or systemic biologics for extensive moderate-to-severe disease.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # BEN_OTH: Other Benign Lesions
    # =========================================================================
    {
        "id": "kb_benoth_typical_look",
        "class_code": "BEN_OTH",
        "section": "typical_look",
        "text": "Other benign cutaneous lesions encompass acrochordons (skin tags), epidermoid cysts, lipomas, fibrous papules, and milia. These typically present as soft, flesh-colored, yellowish, or light-brown circumscribed papules, nodules, or sessile growths that remain stable over prolonged periods.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 115 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_benoth_dermoscopic_features",
        "class_code": "BEN_OTH",
        "section": "dermoscopic_features",
        "text": "Dermoscopy reveals regular, non-atypical architecture with complete absence of malignant criteria. Epidermoid cysts display a central punctum and ivory-white homogeneous coloration; fibrous papules display stable hairpin or comma-like regular vessels without polymorphous vessel atypia.",
        "source": "Kittler H et al. Dermatoscopy: Pattern Analysis, 2nd Ed (Facultas, 2016)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_benoth_differential",
        "class_code": "BEN_OTH",
        "section": "differential",
        "text": "Differential diagnosis includes amelanotic melanoma, early basal cell carcinoma, intradermal nevus, and neurofibroma. Lack of ulceration, arborizing vessels, or atypical pigment structures assists differentiation from cutaneous neoplasms.",
        "source": "Kittler H et al. Dermatoscopy, 2nd Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_benoth_risk_factors",
        "class_code": "BEN_OTH",
        "section": "risk_factors",
        "text": "Predisposing elements include mechanical friction, obesity, metabolic syndrome (associated with multiple acrochordons), chronological aging, and familial genetic factors.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_benoth_red_flags",
        "class_code": "BEN_OTH",
        "section": "red_flags",
        "text": "Rapid change in size, spontaneous recurrent bleeding, sudden onset of firm fixation to underlying deep fascia, or localized pain warrants diagnostic biopsy to rule out soft tissue sarcomas or metastatic cutaneous involvement.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_benoth_darker_skin_nuance",
        "class_code": "BEN_OTH",
        "section": "darker_skin_nuance",
        "text": "In darker skin phototypes, benign cutaneous papules (such as acrochordons and epidermoid cysts) frequently acquire substantial secondary hyperpigmentation, appearing deep brown or dark gray. Overzealous surgical removal carries a recognized risk of inducing keloid formation or permanent hypopigmentation.",
        "source": "Dadzie OE et al. Ethnic Dermatology: Principles and Practice (Wiley, 2013)",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_benoth_next_steps",
        "class_code": "BEN_OTH",
        "section": "next_steps",
        "text": "Clinical reassurance is the primary approach for asymptomatic benign lesions. Elective removal via simple snip excision, light electrodessication, or conservative surgical excision can be considered if lesions cause friction irritation.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },

    # =========================================================================
    # MAL_OTH: Other Malignant Lesions
    # =========================================================================
    {
        "id": "kb_maloth_typical_look",
        "class_code": "MAL_OTH",
        "section": "typical_look",
        "text": "Other malignant skin neoplasms encompass cutaneous T-cell lymphoma (mycosis fungoides), dermatofibrosarcoma protuberans (DFSP), atypical fibroxanthoma (AFX), Merkel cell carcinoma, and cutaneous metastases. These present as progressive, firm, violaceous, or red-brown indurated nodules, infiltrating plaques, or ulcerated masses.",
        "source": "Bolognia JL et al. Dermatology, 4th Ed, Ch. 120 (Elsevier, 2018)",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_maloth_dermoscopic_features",
        "class_code": "MAL_OTH",
        "section": "dermoscopic_features",
        "text": "Dermoscopy characteristically displays prominent polymorphous atypical vessels (corkscrew, serpentine, looped, dotted), structureless shiny white areas and crystalline lines under polarized light, central necrosis, and irregular ulcerations lacking typical melanocytic or basaloid structures.",
        "source": "Lallas A et al. Dermoscopy of Rare Skin Tumours. Eur J Dermatol 2014; 24(6):643-651",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_maloth_differential",
        "class_code": "MAL_OTH",
        "section": "differential",
        "text": "Differential diagnoses include nodular melanoma, poorly differentiated squamous cell carcinoma, deep fungal granulomas, pyogenic granuloma, and benign fibrous histiocytomas. The rapidly enlarging nature and polymorphous vascularity warrant prompt histological clarification.",
        "source": "Lallas A et al. Eur J Dermatol 2014",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_maloth_risk_factors",
        "class_code": "MAL_OTH",
        "section": "risk_factors",
        "text": "Advanced age, extensive chronic immunosuppression (solid organ transplantation, hematologic malignancies), previous ionizing radiation therapy exposure, and Merkel cell polyomavirus infection are recognized risk drivers.",
        "source": "NCCN Guidelines: Merkel Cell Carcinoma, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_maloth_red_flags",
        "class_code": "MAL_OTH",
        "section": "red_flags",
        "text": "Rapid doubling in tumor size over weeks, deep subcutaneous induration, regional lymphadenopathy, spontaneous tissue necrosis, or satellite cutaneous nodule formation represents critical red flags requiring emergent oncology evaluation.",
        "source": "NCCN Guidelines: Merkel Cell Carcinoma, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    },
    {
        "id": "kb_maloth_darker_skin_nuance",
        "class_code": "MAL_OTH",
        "section": "darker_skin_nuance",
        "text": "In darker skin types, rare cutaneous malignancies such as dermatofibrosarcoma protuberans (DFSP) or cutaneous T-cell lymphoma are frequently misdiagnosed as benign keloids, dermatofibromas, or stubborn eczema, leading to diagnostic delays of months to years.",
        "source": "Gloster HM, Neal K. Skin cancer in skin of color. JAAD 2006; 55:741-760",
        "tone_groups_relevant": ["G3"]
    },
    {
        "id": "kb_maloth_next_steps",
        "class_code": "MAL_OTH",
        "section": "next_steps",
        "text": "Immediate urgent referral to a multidisciplinary cutaneous oncology team is essential. Urgent deep incisional or punch biopsy reaching subcutaneous tissue with immunohistochemical staining (CK20, CD34, Melan-A, S100) is required for definitive histogenetic typing.",
        "source": "NCCN Guidelines: Merkel Cell Carcinoma, Version 1.2024",
        "tone_groups_relevant": ["G1", "G2", "G3"]
    }
]


def build_kb():
    KB_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KB_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in CHUNKS:
            # Validate schema
            assert "id" in chunk
            assert "class_code" in chunk
            assert chunk["section"] in [
                "typical_look", "dermoscopic_features", "differential",
                "risk_factors", "red_flags", "darker_skin_nuance", "next_steps"
            ]
            assert len(chunk["text"]) > 20
            assert len(chunk["source"]) > 5
            assert isinstance(chunk["tone_groups_relevant"], list)
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✅ Successfully compiled {len(CHUNKS)} high-quality knowledge base chunks to {KB_OUTPUT_PATH}")


if __name__ == "__main__":
    build_kb()
