"""
VISHWAS Application Internationalization & Strings Dictionary
Provides bilingual UI strings (English & Hindi) for Simple Mode,
glossary terms, medical condition descriptions, and plain-language clinical advice.
Designed to easily add Kannada ('kn') or additional languages.
"""

from __future__ import annotations
from typing import Dict, Any

STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        # App identity
        "app_title": "VISHWAS (विश्वास): Skin AI Check",
        "app_subtitle_simple": "A research prototype for equitable, skin-tone-aware dermatological health screening.",
        "app_subtitle_expert": "Certified 5-Fold Cross-Validation, Multimodal Fusion (Clinical + Dermoscopic + Tabular), and Subgroup Fairness Auditing.",
        "mode_selector": "Interface Mode",
        "mode_simple": "Simple (Guided)",
        "mode_expert": "Expert (Technical)",
        "language_selector": "Language / भाषा",
        "privacy_badge": "🔒 Private & Secure: Your photos are processed in memory and never stored on disk or servers.",
        "medical_disclaimer": "⚠️ Research prototype for education, not a medical device. Always consult a qualified doctor or dermatologist for medical diagnosis.",
        
        # Navigation tabs
        "tab_check": "🔬 Check a Skin Spot",
        "tab_fairness": "⚖️ How Fair Is It?",
        "tab_reliability": "📈 How Reliable Is It?",
        "tab_explorer": "🗂️ Data Explorer",
        "tab_about": "📚 About the Data",

        # Step 1: Photos
        "step1_title": "Step 1: Add photos of the skin spot",
        "step1_desc": "Upload photos of the mole, rash, or skin spot. Adding both a close-up and a magnified photo gives the most reliable results.",
        "label_closeup": "1. Close-up photo of the skin spot",
        "help_closeup": "Take a clear photo with your phone camera from about 10-15 cm away, focusing directly on the spot.",
        "label_dermo": "2. Magnified (dermoscope) photo, if you have one",
        "help_dermo": "A magnified, well-lit photo taken with a dermatoscope lens or macro attachment, if available.",
        "take_photo_btn": "📸 Use Camera to Take Photo",
        "camera_closeup": "Capture Close-up Photo",
        "camera_dermo": "Capture Magnified Photo",
        "single_photo_notice": "ℹ️ Note: You have provided one photo. The AI will still analyze it, but results are generally more reliable when both close-up and magnified views are provided.",
        "photo_tips_title": "💡 Tips for taking a good photo",
        "photo_tips_content": """• **Good lighting**: Use bright, indirect natural daylight or a clean flash.\n• **In focus**: Tap your screen on the spot so the image is sharp, not blurry.\n• **Fill the frame**: Center the spot so it takes up 30-50% of the image.\n• **No beauty filters**: Turn off skin-smoothing or color-altering filters.\n• **Size reference**: If possible, place a coin or small ruler next to the spot for scale.""",

        # Quality warnings
        "warn_blurry": "⚠️ This photo looks blurry. Retaking it in sharp focus will yield a much more accurate estimate.",
        "warn_too_dark": "⚠️ This photo is very dark. Retaking it under brighter lighting will improve accuracy.",
        "warn_too_bright": "⚠️ This photo is too bright / overexposed or has severe glare. Please retake it without direct reflection.",
        "warn_low_res": "⚠️ The photo resolution is quite low. A clearer, higher-resolution photo is recommended.",

        # Step 2: Patient details
        "step2_title": "Step 2: About the patient (optional)",
        "step2_desc": "More details can make the result more accurate. All fields are completely optional.",
        "label_age": "Approximate Age (years)",
        "help_age": "Patient age helps the AI distinguish between age-associated benign spots and other conditions.",
        "dont_know_age": "I don't know the exact age",
        "label_sex": "Sex Assigned at Birth",
        "help_sex": "Some skin conditions appear with differing prevalence across biological sexes.",
        "sex_male": "Male",
        "sex_female": "Female",
        "sex_unknown": "Prefer not to say",
        "label_site": "Where on the body is this spot located?",
        "help_site": "Anatomical location provides critical clinical context for dermatological differential diagnosis.",
        "site_trunk": "🫁 Chest, Back, or Stomach",
        "site_head_neck": "👤 Head, Face, or Neck",
        "site_lower_ext": "🦵 Leg, Thigh, or Hip",
        "site_upper_ext": "💪 Arm or Shoulder",
        "site_hand": "✋ Hand",
        "site_foot": "🦶 Foot",
        "site_genital": "🩲 Groin or Genital region",
        "site_unknown": "❓ Other / Not sure",

        # Step 3: Result
        "step3_title": "Step 3: Result & What You Can Do",
        "result_heading": "AI Clinical Assessment",
        "most_likely": "Most Likely Finding",
        "confidence_level": "AI Confidence",
        "conf_high": "High Confidence",
        "conf_med": "Medium Confidence",
        "conf_low": "Low Confidence (Needs Clinical Review)",
        "ai_estimate_tag": "ℹ️ This is an AI estimate, not a medical diagnosis.",
        "rerun_btn": "🔄 Re-run Analysis",
        "start_over_btn": "🔄 Start Over",
        "next_steps_title": "What you can do next",
        "advice_malignant": "🚨 **Please show this to a doctor soon.** The AI detected features that require clinical evaluation and possible biopsy.",
        "advice_premalignant": "⚠️ **Please show this to a doctor soon.** This spot shows pre-cancerous or atypical features that may benefit from preventative treatment.",
        "advice_benign": "✅ **Usually harmless, but see a doctor if it changes or worries you.** Continue routine self-checks.",
        "advice_nontumour": "🩺 **May need treatment, ask a doctor.** This appears to be an inflammatory or vascular condition rather than a tumor.",
        "clinical_review_banner": "🚨 **Recommendation for Professional Review**: Due to elevated risk features or low model certainty, we strongly recommend having this spot examined in person by a healthcare provider.",
        "top3_title": "Top Possibilities Considered by the AI",
        "show_details": "🔬 Show technical & multi-model details",
        "download_pdf_btn": "📥 Download My Result Summary (PDF)",
        "download_json_btn": "📥 Download My Result (JSON)",

        # Empty state & examples
        "how_it_works_title": "👋 How It Works (3 Quick Steps)",
        "how_it_works_step1": "**1. Add photos**: Take or upload a close-up picture of the skin lesion.",
        "how_it_works_step2": "**2. Add details (optional)**: Provide age or body location to assist accuracy.",
        "how_it_works_step3": "**3. View result**: Instantly view the estimated condition, likelihood, and next steps.",
        "try_example_title": "Or try an example case:",
        "example_mel": "Example 1: Melanoma (Darker skin tone)",
        "example_bcc": "Example 2: Basal Cell (Lighter skin tone)",
        "example_nv": "Example 3: Normal Mole (Medium skin tone)",
        "dismiss_banner": "✕ Dismiss",

        # Progress messages
        "progress_looking": "Looking at your photos and checking quality...",
        "progress_models": "Analyzing across 5 trained deep-learning models on Apple Silicon...",
        "progress_preparing": "Synthesizing multimodal results and generating explanation...",
        "toast_done": "✅ Analysis completed successfully!",

        # Session history
        "history_title": "🕒 Recent checks in this session",
        "history_empty": "No recent checks yet.",
        "restore_btn": "Restore",

        # Glossary
        "glossary_title": "📖 Plain-Language Glossary",
        "glossary_lesion": "**Lesion**: Any abnormal spot, bump, mole, or patch on the skin.",
        "glossary_dermoscopy": "**Dermoscopy**: A specialized magnifier with polarized light used by dermatologists to see under the skin surface.",
        "glossary_melanoma": "**Melanoma**: A serious form of skin cancer arising from pigment cells (melanocytes). Early detection is critical.",
        "glossary_bcc": "**Basal Cell Carcinoma (BCC)**: The most common type of skin cancer; usually slow-growing and treatable.",
        "glossary_f1": "**Macro-F1 Score**: A balanced metric (0 to 1) of diagnostic accuracy that treats rare and common diseases equally.",
        "glossary_calibration": "**Calibration**: A measure of whether the AI's percentage confidence matches real-world clinical certainty.",
        "glossary_tone_group": "**Tone Group**: A grouping of skin phototypes (G1 = light, G2 = medium, G3 = deep) used to audit fairness.",

        # Chart explainers
        "chart_explain_fairness": "💡 **What does this chart mean?** This chart compares performance across people with lighter (G1), medium (G2), and darker (G3) skin tones. A fair model achieves equal accuracy across all tone groups without leaving any group behind.",
        "chart_explain_calibration": "💡 **What does this chart mean?** A well-calibrated AI only reports 90% confidence when it is truly right 90% of the time. Points closer to the dotted diagonal line indicate more trustworthy probability estimates.",
        "chart_explain_confusion": "💡 **What does this chart mean?** This matrix shows which diseases the AI correctly identified along the diagonal, and which ones it occasionally confused with similar-looking skin spots.",
    },

    "hi": {
        # App identity
        "app_title": "VISHWAS (विश्वास): त्वचा स्वास्थ्य जांच",
        "app_subtitle_simple": "त्वचा के रंग-भेद रहित, निष्पक्ष और सटीक त्वचा स्वास्थ्य मूल्यांकन के लिए एक शोध प्रोटोटाइप।",
        "app_subtitle_expert": "प्रमाणित 5-फ़ोल्ड क्रॉस-सत्यापन, मल्टीमॉडल फ़्यूज़न और उपसमूह निष्पक्षता ऑडिटिंग प्रणाली।",
        "mode_selector": "इंटरफ़ेस मोड",
        "mode_simple": "सरल (मार्गदर्शित)",
        "mode_expert": "विशेषज्ञ (तकनीकी)",
        "language_selector": "भाषा / Language",
        "privacy_badge": "🔒 पूर्णतः निजी एवं सुरक्षित: आपकी तस्वीरें केवल मेमोरी में संसाधित होती हैं और कभी भी सहेजी नहीं जातीं।",
        "medical_disclaimer": "⚠️ यह शोध और शिक्षा के लिए एक प्रोटोटाइप है, कोई प्रमाणित चिकित्सा उपकरण नहीं। चिकित्सीय सलाह के लिए हमेशा डॉक्टर से संपर्क करें।",
        
        # Navigation tabs
        "tab_check": "🔬 त्वचा के धब्बे की जांच",
        "tab_fairness": "⚖️ मॉडल कितना निष्पक्ष है?",
        "tab_reliability": "📈 मॉडल कितना विश्वसनीय है?",
        "tab_explorer": "🗂️ डेटा एक्सप्लोरर",
        "tab_about": "📚 डेटासेट परिचय",

        # Step 1: Photos
        "step1_title": "चरण 1: त्वचा के धब्बे की तस्वीरें जोड़ें",
        "step1_desc": "तिल, चकत्ते या धब्बे की तस्वीरें अपलोड करें। क्लोज़-अप और आवर्धित (डर्मोस्कोपिक) दोनों तस्वीरें देने से सबसे सटीक परिणाम मिलता है।",
        "label_closeup": "1. त्वचा के धब्बे की क्लोज़-अप तस्वीर",
        "help_closeup": "अपने फ़ोन कैमरे से लगभग 10-15 सेमी की दूरी से धब्बे पर साफ़ फ़ोकस करके तस्वीर लें।",
        "label_dermo": "2. आवर्धित (डर्मोस्कोप) तस्वीर, यदि आपके पास हो",
        "help_dermo": "डर्मेटोस्कोप या मैक्रो लेंस से ली गई आवर्धित और अच्छी रोशनी वाली तस्वीर (वैकल्पिक)।",
        "take_photo_btn": "📸 कैमरा का उपयोग करके फ़ोटो खींचें",
        "camera_closeup": "क्लोज़-अप फ़ोटो खींचें",
        "camera_dermo": "आवर्धित फ़ोटो खींचें",
        "single_photo_notice": "ℹ️ सूचना: आपने केवल एक तस्वीर दी है। AI इसका विश्लेषण करेगा, लेकिन दोनों कोणों की तस्वीरें होने पर परिणाम अधिक सटीक होते हैं।",
        "photo_tips_title": "💡 अच्छी तस्वीर लेने के महत्वपूर्ण सुझाव",
        "photo_tips_content": """• **अच्छी रोशनी**: प्राकृतिक दिन के उजाले या साफ़ फ़्लैश का प्रयोग करें।\n• **स्पष्ट फ़ोकस**: स्क्रीन पर धब्बे को छूकर फ़ोकस करें ताकि तस्वीर धुंधली न हो।\n• **तस्वीर में धब्बे का आकार**: धब्बा फ़्रेम के 30-50% हिस्से में दिखना चाहिए।\n• **कोई फ़िल्टर नहीं**: त्वचा को साफ़ करने वाले ब्यूटी फ़िल्टर बंद रखें।\n• **आकार का संदर्भ**: यदि संभव हो तो धब्बे के पास एक सिक्का या छोटा पैमाना रखें।""",

        # Quality warnings
        "warn_blurry": "⚠️ यह तस्वीर धुंधली लग रही है। साफ़ फ़ोकस में दोबारा फ़ोटो लेने से अधिक सटीक परिणाम मिलेगा।",
        "warn_too_dark": "⚠️ यह तस्वीर बहुत अंधेरी है। कृपया बेहतर रोशनी में दोबारा तस्वीर लें।",
        "warn_too_bright": "⚠️ इस तस्वीर में अत्यधिक चमक या चकाचौंध है। कृपया बिना प्रतिबिंब के दोबारा फ़ोटो लें।",
        "warn_low_res": "⚠️ तस्वीर का रिज़ॉल्यूशन बहुत कम है। कृपया अधिक स्पष्ट तस्वीर अपलोड करें।",

        # Step 2: Patient details
        "step2_title": "चरण 2: रोगी का विवरण (वैकल्पिक)",
        "step2_desc": "विवरण देने से परिणाम अधिक सटीक हो सकता है। सभी फ़ील्ड पूर्णतः वैकल्पिक हैं।",
        "label_age": "अनुमानित आयु (वर्ष)",
        "help_age": "उम्र से AI को उम्र-संबंधी सामान्य तिलों और गंभीर स्थितियों में अंतर करने में मदद मिलती है।",
        "dont_know_age": "मुझे सटीक उम्र नहीं पता",
        "label_sex": "जन्म के समय दर्ज लिंग",
        "help_sex": "कुछ त्वचा रोग जैविक लिंग के आधार पर भिन्न व्यापकता दर्शाते हैं।",
        "sex_male": "पुरुष",
        "sex_female": "महिला",
        "sex_unknown": "बताना नहीं चाहते",
        "label_site": "शरीर के किस हिस्से पर यह धब्बा स्थित है?",
        "help_site": "शरीर पर धब्बे का स्थान त्वचा रोग निदान के लिए महत्वपूर्ण नैदानिक संदर्भ प्रदान करता है।",
        "site_trunk": "🫁 सीना, पीठ या पेट",
        "site_head_neck": "👤 सिर, चेहरा या गर्दन",
        "site_lower_ext": "🦵 पैर, जांघ या कूल्हा",
        "site_upper_ext": "💪 हाथ या कंधा",
        "site_hand": "✋ हथेली",
        "site_foot": "🦶 पैर का पंजा",
        "site_genital": "🩲 जननांग क्षेत्र",
        "site_unknown": "❓ अन्य / निश्चित नहीं",

        # Step 3: Result
        "step3_title": "चरण 3: परिणाम और आगे क्या करें",
        "result_heading": "AI नैदानिक मूल्यांकन",
        "most_likely": "सर्वाधिक संभावित स्थिति",
        "confidence_level": "AI विश्वास स्तर",
        "conf_high": "उच्च विश्वास",
        "conf_med": "मध्यम विश्वास",
        "conf_low": "कम विश्वास (डॉक्टर की जांच आवश्यक)",
        "ai_estimate_tag": "ℹ️ यह एक AI अनुमान है, चिकित्सकीय निदान नहीं।",
        "rerun_btn": "🔄 पुनः विश्लेषण करें",
        "start_over_btn": "🔄 नए सिरे से शुरू करें",
        "next_steps_title": "आप आगे क्या कर सकते हैं",
        "advice_malignant": "🚨 **कृपया शीघ्र ही किसी डॉक्टर या त्वचा विशेषज्ञ को दिखाएं।** AI ने ऐसे लक्षण पाए हैं जिनकी डॉक्टरी जांच और बायोप्सी आवश्यक हो सकती है।",
        "advice_premalignant": "⚠️ **कृपया जल्द ही डॉक्टर से परामर्श लें।** यह धब्बा पूर्व-कैंसर या असामान्य वृद्धि दर्शाता है, जिसका समय पर उपचार लाभदायक है।",
        "advice_benign": "✅ **यह आमतौर पर हानिरहित है, लेकिन यदि इसमें बदलाव, वृद्धि या रक्तस्राव हो तो डॉक्टर को दिखाएं।** नियमित निगरानी जारी रखें।",
        "advice_nontumour": "🩺 **उपचार की आवश्यकता हो सकती है, डॉक्टर से पूछें।** यह गांठ या ट्यूमर के बजाय सूजन या संवहनी स्थिति प्रतीत होती है।",
        "clinical_review_banner": "🚨 **पेशेवर समीक्षा की सिफारिश**: उच्च जोखिम वाले लक्षणों या मॉडल की कम निश्चितता के कारण, हम डॉक्टर से प्रत्यक्ष जांच कराने की सलाह देते हैं।",
        "top3_title": "AI द्वारा विचार की गई शीर्ष 3 संभावनाएं",
        "show_details": "🔬 तकनीकी और मॉडल विवरण देखें",
        "download_pdf_btn": "📥 परिणाम सारांश डाउनलोड करें (PDF)",
        "download_json_btn": "📥 परिणाम डाउनलोड करें (JSON)",

        # Empty state & examples
        "how_it_works_title": "👋 यह कैसे काम करता है (3 आसान चरण)",
        "how_it_works_step1": "**1. फ़ोटो जोड़ें**: त्वचा के धब्बे की साफ़ क्लोज़-अप तस्वीर लें या अपलोड करें।",
        "how_it_works_step2": "**2. विवरण जोड़ें (वैकल्पिक)**: सटीकता बढ़ाने के लिए उम्र या शरीर का स्थान बताएं।",
        "how_it_works_step3": "**3. परिणाम देखें**: तुरंत संभावित स्थिति, विश्वास स्तर और अगला कदम जानें।",
        "try_example_title": "या कोई उदाहरण आज़माएं:",
        "example_mel": "उदाहरण 1: मेलेनोमा (गहरा त्वचा रंग)",
        "example_bcc": "उदाहरण 2: बेसल सेल (हल्का त्वचा रंग)",
        "example_nv": "उदाहरण 3: सामान्य तिल (मध्यम त्वचा रंग)",
        "dismiss_banner": "✕ बंद करें",

        # Progress messages
        "progress_looking": "आपकी तस्वीरों की गुणवत्ता जांची जा रही है...",
        "progress_models": "5 डीप-लर्निंग मॉडलों द्वारा विश्लेषण किया जा रहा है...",
        "progress_preparing": "परिणाम संकलित कर अंतिम रिपोर्ट तैयार की जा रही है...",
        "toast_done": "✅ विश्लेषण सफलतापूर्वक पूर्ण हुआ!",

        # Session history
        "history_title": "🕒 इस सत्र की हालिया जांचें",
        "history_empty": "अभी कोई हालिया जांच नहीं है।",
        "restore_btn": "पुनर्स्थापित करें",

        # Glossary
        "glossary_title": "📖 सरल शब्दावली",
        "glossary_lesion": "**घाव (Lesion)**: त्वचा पर कोई भी असामान्य धब्बा, तिल, उभार या चकत्ता।",
        "glossary_dermoscopy": "**डर्मोस्कोपी (Dermoscopy)**: त्वचा विशेषज्ञों द्वारा त्वचा की सतह के नीचे देखने के लिए प्रयुक्त विशेष आवर्धक लेंस।",
        "glossary_melanoma": "**मेलेनोमा (Melanoma)**: त्वचा कैंसर का एक गंभीर रूप जो वर्णक कोशिकाओं से उत्पन्न होता है। शीघ्र पहचान अत्यंत आवश्यक है।",
        "glossary_bcc": "**बेसल सेल कार्सिनोमा (BCC)**: त्वचा कैंसर का सबसे आम प्रकार; आमतौर पर धीमा बढ़ता है और आसानी से उपचार योग्य है।",
        "glossary_f1": "**मैक्रो-F1 स्कोर**: सटीकता का एक संतुलित पैमाना (0 से 1) जो दुर्लभ और सामान्य बीमारियों को समान महत्व देता है।",
        "glossary_calibration": "**अंशांकन (Calibration)**: यह माप कि AI का प्रतिशत विश्वास वास्तविक नैदानिक निश्चितता से कितना मेल खाता है।",
        "glossary_tone_group": "**स्किन टोन समूह**: त्वचा के रंगों का वर्गीकरण (G1 = हल्का, G2 = मध्यम, G3 = गहरा) जो निष्पक्षता जांचने के काम आता है।",

        # Chart explainers
        "chart_explain_fairness": "💡 **इस चार्ट का क्या अर्थ है?** यह चार्ट हल्के (G1), मध्यम (G2) और गहरे (G3) त्वचा के रंग वाले लोगों में AI की सटीकता की तुलना करता है। एक निष्पक्ष मॉडल सभी समूहों में समान रूप से सही काम करता है।",
        "chart_explain_calibration": "💡 **इस चार्ट का क्या अर्थ है?** एक अच्छी तरह कैलिब्रेटेड AI केवल तभी 90% विश्वास दिखाता है जब वह वास्तव में 90% समय सही हो। बिंदीदार रेखा के करीब होना बेहतर विश्वसनीयता दर्शाता है।",
        "chart_explain_confusion": "💡 **इस चार्ट का क्या अर्थ है?** यह मैट्रिक्स दिखाता है कि AI ने किन रोगों को सही पहचाना, और किन रोगों में वह समान दिखने वाले धब्बों के बीच भ्रमित हुआ।",
    }
}

# Detailed descriptions of the 11 classes with bilingual support
CONDITION_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "AKIEC": {
        "en_name": "Actinic Keratosis / Bowen's Disease",
        "hi_name": "एक्टिनिक केराटोसिस / बोवेन रोग",
        "en_desc": "A rough, scaly patch on sun-damaged skin considered pre-cancerous, which can occasionally develop into squamous cell cancer if left untreated.",
        "hi_desc": "धूप से क्षतिग्रस्त त्वचा पर एक खुरदरा, पपड़ीदार धब्बा जो पूर्व-कैंसर माना जाता है और अनुपचारित रहने पर बढ़ सकता है।",
        "category": "premalignant",
    },
    "BCC": {
        "en_name": "Basal Cell Carcinoma",
        "hi_name": "बेसल सेल कार्सिनोमा",
        "en_desc": "The most common form of skin cancer. It usually appears as a slightly shiny bump or pink patch, grows slowly, and rarely spreads if treated.",
        "hi_desc": "त्वचा कैंसर का सबसे आम प्रकार। यह अक्सर चमकदार उभार या गुलाबी धब्बे के रूप में दिखता है और उपचार से ठीक हो जाता है।",
        "category": "malignant",
    },
    "BEN_OTH": {
        "en_name": "Other Benign Skin Lesion",
        "hi_name": "अन्य हानिरहित त्वचा उभार",
        "en_desc": "A non-cancerous skin growth such as a skin tag, epidermoid cyst, or solar lentigo that presents no medical danger.",
        "hi_desc": "एक गैर-कैंसरयुक्त हानिरहित त्वचा वृद्धि जैसे मस्सा या सिस्ट जिससे कोई चिकित्सीय खतरा नहीं होता।",
        "category": "benign",
    },
    "BKL": {
        "en_name": "Benign Keratotic Lesion (Seborrheic Keratosis)",
        "hi_name": "सौम्य केराटोटिक घाव (सेबोरहाइक केराटोसिस)",
        "en_desc": "A very common, harmless warty or waxy spot that appears 'stuck onto' the skin as people age.",
        "hi_desc": "उम्र बढ़ने के साथ दिखने वाला एक बहुत ही सामान्य और हानिरहित मस्से जैसा धब्बा जो त्वचा पर चिपका हुआ सा लगता है।",
        "category": "benign",
    },
    "DF": {
        "en_name": "Dermatofibroma",
        "hi_name": "डर्माटोफ़ाइब्रोमा",
        "en_desc": "A firm, harmless small bump commonly found on the legs, often following an insect bite or minor skin injury.",
        "hi_desc": "आमतौर पर पैरों पर कीड़े के काटने या छोटी चोट के बाद बनने वाली एक सख्त, हानिरहित छोटी गांठ।",
        "category": "benign",
    },
    "INF": {
        "en_name": "Inflammatory or Infectious Condition",
        "hi_name": "सूजन या संक्रमण संबंधी स्थिति",
        "en_desc": "A reactive skin condition such as eczema, psoriasis, dermatitis, or a localized bacterial/fungal skin reaction.",
        "hi_desc": "एक्जिमा, सोरायसिस या संक्रमण जैसी सूजनयुक्त त्वचा स्थिति जिसके लिए मलहम या डॉक्टरी उपचार की आवश्यकता हो सकती है।",
        "category": "nontumour",
    },
    "MAL_OTH": {
        "en_name": "Other Rare Malignancy",
        "hi_name": "अन्य दुर्लभ त्वचा दुर्दमता",
        "en_desc": "A less common form of skin cancer (e.g. cutaneous lymphoma, Merkel cell carcinoma) requiring prompt histological examination.",
        "hi_desc": "त्वचा कैंसर का एक कम सामान्य रूप जिसके लिए तत्काल विशेषज्ञ जांच और बायोप्सी की आवश्यकता होती है।",
        "category": "malignant",
    },
    "MEL": {
        "en_name": "Melanoma",
        "hi_name": "मेलेनोमा (गंभीर त्वचा कैंसर)",
        "en_desc": "A high-risk type of skin cancer that begins in pigment-producing cells. Early detection and immediate excision are critical.",
        "hi_desc": "त्वचा कैंसर का एक गंभीर रूप। इसकी शीघ्र पहचान और विशेषज्ञ द्वारा त्वरित उपचार अत्यंत आवश्यक है।",
        "category": "malignant",
    },
    "NV": {
        "en_name": "Melanocytic Nevus (Normal Mole)",
        "hi_name": "मेलेनोसाइटिक नेवस (सामान्य तिल)",
        "en_desc": "A standard, benign collection of pigment cells. Normal moles are completely harmless but should be watched for changes.",
        "hi_desc": "वर्णक कोशिकाओं का एक सामान्य, हानिरहित तिल। यह पूरी तरह सुरक्षित होता है पर किसी भी बदलाव पर ध्यान देना चाहिए।",
        "category": "benign",
    },
    "SCCKA": {
        "en_name": "Squamous Cell Carcinoma / Keratoacanthoma",
        "hi_name": "स्क्वैमस सेल कार्सिनोमा",
        "en_desc": "The second most common skin cancer, often appearing as a firm red nodule or scaly, crusting sore that does not heal.",
        "hi_desc": "दूसरा सबसे आम त्वचा कैंसर, जो अक्सर पपड़ीदार या न भरने वाले घाव के रूप में दिखता है।",
        "category": "malignant",
    },
    "VASC": {
        "en_name": "Vascular Lesion (Cherry Angioma, Pyogenic Granuloma)",
        "hi_name": "संवहनी घाव (रक्त वाहिका उभार)",
        "en_desc": "A benign collection of small blood vessels in the skin, such as a bright red cherry angioma or vascular malformation.",
        "hi_desc": "रक्त वाहिकाओं का एक सामान्य, लाल रंग का हानिरहित उभार जिससे कोई चिकित्सीय खतरा नहीं होता।",
        "category": "nontumour",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Returns the translated string for a key in the specified language, falling back to English."""
    lang_dict = STRINGS.get(lang, STRINGS["en"])
    return lang_dict.get(key, STRINGS["en"].get(key, key))
