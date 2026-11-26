"""
VISHWAS Application Internationalization & Strings Dictionary
Provides multilingual UI strings (English, Hindi, and Kannada) for Simple Mode,
glossary terms, medical condition descriptions, traffic-light triage verdicts,
and plain-language clinical advice.
"""

from __future__ import annotations
import logging
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
        "language_selector": "Language / भाषा / ಭಾಷೆ",
        "privacy_badge": "🔒 Private & Secure: Your photos are processed in memory and never stored on disk or servers.",
        "medical_disclaimer": "⚠️ Research prototype for education, not a medical device. Always consult a qualified doctor or dermatologist for medical diagnosis.",

        # Theme
        "theme_selector": "Theme / Mode",
        "theme_auto": "Auto (System)",
        "theme_light": "Light",
        "theme_dark": "Dark",

        # Navigation tabs
        "tab_check": "🔬 Check a Skin Spot",
        "tab_fairness": "⚖️ How Fair Is It?",
        "tab_reliability": "📈 How Reliable Is It?",
        "tab_explorer": "🗂️ Data Explorer",
        "tab_about": "📚 About the Data",
        "tab_conditions_guide": "🩺 Conditions Guide",

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
        "photo_tips_content": "• **Good lighting**: Use bright, indirect natural daylight or a clean flash.\n• **In focus**: Tap your screen on the spot so the image is sharp, not blurry.\n• **Fill the frame**: Center the spot so it takes up 30-50% of the image.\n• **No beauty filters**: Turn off skin-smoothing or color-altering filters.\n• **Size reference**: If possible, place a coin or small ruler next to the spot for scale.",

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

        # Step 3: Result & Pinned Strip
        "step3_title": "Step 3: Result & What You Can Do",
        "result_heading": "AI Clinical Assessment",
        "most_likely": "Most Likely Finding",
        "most_likely_uncertain": "Most Likely Finding (Uncertain)",
        "confidence_level": "AI Confidence",
        "conf_high": "High Confidence",
        "conf_med": "Medium Confidence",
        "conf_low": "Low Confidence (Needs Clinical Review)",
        "ai_estimate_tag": "ℹ️ This is an AI estimate, not a medical diagnosis.",
        "rerun_btn": "🔄 Re-run Analysis",
        "start_over_btn": "🔄 Start Over",
        "next_steps_title": "What you can do next",
        "advice_malignant": "**Please show this to a doctor soon.** The AI detected features that require clinical evaluation and possible biopsy.",
        "advice_premalignant": "**Please show this to a doctor soon.** This spot shows pre-cancerous or atypical features that may benefit from preventative treatment.",
        "advice_benign": "**Usually harmless, but see a doctor if it changes or worries you.** Continue routine self-checks.",
        "advice_nontumour": "**May need treatment, ask a doctor.** This appears to be an inflammatory or vascular condition rather than a tumor.",
        "clinical_review_banner": "**Recommendation for Professional Review**: Due to elevated risk features or low model certainty, we strongly recommend having this spot examined in person by a healthcare provider.",
        "top3_title": "Top Possibilities Considered by the AI",
        "show_details": "🔬 Show technical & multi-model details",
        "download_pdf_btn": "📥 Download My Result (PDF)",
        "download_json_btn": "📥 Download My Result (JSON)",
        "pdf_fallback_notice": "ℹ️ Note: Diagnostic PDF report is generated in English for clinical documentation clarity.",

        # Triage verdicts (titles without duplicate leading emoji)
        "triage_red_title": "See a doctor soon",
        "triage_red_text": "The AI detected high-risk or pre-cancerous features. Please arrange an in-person clinical check promptly.",
        "triage_amber_title": "Get it checked",
        "triage_amber_text": "The AI found non-benign or inflammatory features, or confidence is modest. An in-person medical check is recommended.",
        "triage_green_title": "Likely harmless, keep an eye on it",
        "triage_green_text": "The spot appears benign and high-risk likelihood is low. Continue routine self-checks and see a doctor if it changes.",
        "triage_grey_title": "Can't tell, retake the photo",
        "triage_grey_text": "Photo quality issues or very low confidence prevent a reliable estimate. Please retake the photo under clear lighting.",

        # Modal Popup & Pinned Strip
        "popup_title": "Diagnostic Assessment Result",
        "popup_close_btn": "✕ Close",
        "view_details_btn": "🔍 View Details & Result",
        "result_strip_title": "Latest Assessment Summary",
        "evidence_line": "In testing, when the model was this confident, it was right about {accuracy}% of the time.",

        # Empty state & examples
        "how_it_works_title": "👋 How It Works (3 Quick Steps)",
        "how_it_works_step1": "1. Add photos: Take or upload a close-up picture of the skin lesion.",
        "how_it_works_step2": "2. Add details (optional): Provide age or body location to assist accuracy.",
        "how_it_works_step3": "3. View result: Instantly view the estimated condition, likelihood, and next steps.",
        "try_example_title": "Or try an example case:",
        "example_mel": "Example 1: Melanoma (Darker skin tone)",
        "example_bcc": "Example 2: Basal Cell (Lighter skin tone)",
        "example_nv": "Example 3: Normal Mole (Medium skin tone)",
        "example_badge": "🏷️ Example Case",
        "user_photo_badge": "📸 User Photo",
        "dismiss_banner": "✕ Dismiss",

        # Progress messages
        "progress_looking": "Looking at your photos and checking quality...",
        "progress_models": "Analyzing across 5 trained deep-learning models on Apple Silicon...",
        "progress_preparing": "Synthesizing multimodal results and generating explanation...",
        "toast_done": "✅ Analysis completed successfully!",

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
        "language_selector": "भाषा / Language / ಭಾಷೆ",
        "privacy_badge": "🔒 पूर्णतः निजी एवं सुरक्षित: आपकी तस्वीरें केवल मेमोरी में संसाधित होती हैं और कभी भी सहेजी नहीं जातीं।",
        "medical_disclaimer": "⚠️ यह शोध और शिक्षा के लिए एक प्रोटोटाइप है, कोई प्रमाणित चिकित्सा उपकरण नहीं। चिकित्सीय सलाह के लिए हमेशा डॉक्टर से संपर्क करें।",

        # Theme
        "theme_selector": "थीम / Mode",
        "theme_auto": "ऑटो (सिस्टम अनुसार)",
        "theme_light": "लाइट (हल्का)",
        "theme_dark": "डार्क (गहरा)",

        # Navigation tabs
        "tab_check": "🔬 त्वचा के धब्बे की जांच",
        "tab_fairness": "⚖️ मॉडल कितना निष्पक्ष है?",
        "tab_reliability": "📈 मॉडल कितना विश्वसनीय है?",
        "tab_explorer": "🗂️ डेटा एक्सप्लोरर",
        "tab_about": "📚 डेटासेट परिचय",
        "tab_conditions_guide": "🩺 रोग मार्गदर्शिका",

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
        "photo_tips_content": "• **अच्छी रोशनी**: प्राकृतिक दिन के उजाले या साफ़ फ़्लैश का प्रयोग करें।\n• **स्पष्ट फ़ोकस**: स्क्रीन पर धब्बे को छूकर फ़ोकस करें ताकि तस्वीर धुंधली न हो।\n• **तस्वीर में धब्बे का आकार**: धब्बा फ़्रेम के 30-50% हिस्से में दिखना चाहिए।\n• **कोई फ़िल्टर नहीं**: त्वचा को साफ़ करने वाले ब्यूटी फ़िल्टर बंद रखें।\n• **आकार का संदर्भ**: यदि संभव हो तो धब्बे के पास एक सिक्का या छोटा पैमाना रखें।",

        # Quality warnings
        "warn_blurry": "⚠️ यह तस्वीर धुंधली लग रही है। साफ़ फ़ोकस में दोबारा फ़ोटो लेने से अधिक सटीक परिणाम मिलेगा।",
        "warn_too_dark": "⚠️ यह तस्वीर बहुत अंधेरी है। कृपया बेहतर रोशनी में दोबारा तस्वीर लें।",
        "warn_too_bright": "⚠️ इस तस्वीर में अत्यधिक रोशनी / चमक या चकाचौंध है। कृपया बिना प्रतिबिंब के दोबारा फ़ोटो लें।",
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

        # Step 3: Result & Pinned Strip
        "step3_title": "चरण 3: परिणाम और आगे क्या करें",
        "result_heading": "AI नैदानिक मूल्यांकन",
        "most_likely": "सर्वाधिक संभावित स्थिति",
        "most_likely_uncertain": "सर्वाधिक संभावित स्थिति (अनिश्चित)",
        "confidence_level": "AI विश्वास स्तर",
        "conf_high": "उच्च विश्वास",
        "conf_med": "मध्यम विश्वास",
        "conf_low": "कम विश्वास (डॉक्टर की जांच आवश्यक)",
        "ai_estimate_tag": "ℹ️ यह एक AI अनुमान है, चिकित्सकीय निदान नहीं।",
        "rerun_btn": "🔄 पुनः विश्लेषण करें",
        "start_over_btn": "🔄 नए सिरे से शुरू करें",
        "next_steps_title": "आप आगे क्या कर सकते हैं",
        "advice_malignant": "**कृपया शीघ्र ही किसी डॉक्टर या त्वचा विशेषज्ञ को दिखाएं।** AI ने ऐसे लक्षण पाए हैं जिनकी डॉक्टरी जांच और बायोप्सी आवश्यक हो सकती है।",
        "advice_premalignant": "**कृपया जल्द ही डॉक्टर से परामर्श लें।** यह धब्बा पूर्व-कैंसर या असामान्य वृद्धि दर्शाता है, जिसका समय पर उपचार लाभदायक है।",
        "advice_benign": "**यह आमतौर पर हानिरहित है, लेकिन यदि इसमें बदलाव, वृद्धि या रक्तस्राव हो तो डॉक्टर को दिखाएं।** नियमित निगरानी जारी रखें।",
        "advice_nontumour": "**उपचार की आवश्यकता हो सकती है, डॉक्टर से पूछें।** यह गांठ या ट्यूमर के बजाय सूजन या संवहनी स्थिति प्रतीत होती है।",
        "clinical_review_banner": "**पेशेवर समीक्षा की सिफारिश**: उच्च जोखिम वाले लक्षणों या मॉडल की कम निश्चितता के कारण, हम डॉक्टर से प्रत्यक्ष जांच कराने की सलाह देते हैं।",
        "top3_title": "AI द्वारा विचार की गई शीर्ष 3 संभावनाएं",
        "show_details": "🔬 तकनीकी और मॉडल विवरण देखें",
        "download_pdf_btn": "📥 परिणाम सारांश डाउनलोड करें (PDF)",
        "download_json_btn": "📥 परिणाम डाउनलोड करें (JSON)",
        "pdf_fallback_notice": "ℹ️ सूचना: नैदानिक स्पष्टता के लिए PDF रिपोर्ट अंग्रेजी में तैयार की जाती है।",

        # Triage verdicts (without duplicate emoji)
        "triage_red_title": "शीघ्र डॉक्टर को दिखाएं",
        "triage_red_text": "AI ने उच्च जोखिम या पूर्व-कैंसर वाले लक्षण पाए हैं। कृपया तुरंत किसी त्वचा विशेषज्ञ से प्रत्यक्ष जांच कराएं।",
        "triage_amber_title": "जांच करवाएं",
        "triage_amber_text": "AI ने सूजन या असामान्य लक्षण पाए हैं, अथवा विश्वास मध्यम है। डॉक्टर से जांच करवाने की सलाह दी जाती है।",
        "triage_green_title": "हानिरहित प्रतीत होता है, निगरानी रखें",
        "triage_green_text": "यह धब्बा सुरक्षित लगता है और गंभीर जोखिम कम है। नियमित निगरानी रखें और बदलाव दिखने पर डॉक्टर को दिखाएं।",
        "triage_grey_title": "स्पष्ट नहीं है, दोबारा तस्वीर लें",
        "triage_grey_text": "तस्वीर की गुणवत्ता में कमी या बहुत कम विश्वास के कारण अनुमान नहीं लगाया जा सका। कृपया स्पष्ट रोशनी में दोबारा फ़ोटो लें।",

        # Modal Popup & Pinned Strip
        "popup_title": "नैदानिक मूल्यांकन परिणाम",
        "popup_close_btn": "✕ बंद करें",
        "view_details_btn": "🔍 विवरण और परिणाम देखें",
        "result_strip_title": "ताज़ा मूल्यांकन सारांश",
        "evidence_line": "परीक्षण में, जब मॉडल इतना आश्वस्त था, तब वह लगभग {accuracy}% बार सही था।",

        # Empty state & examples
        "how_it_works_title": "👋 यह कैसे काम करता है (3 आसान चरण)",
        "how_it_works_step1": "1. फ़ोटो जोड़ें: त्वचा के धब्बे की साफ़ क्लोज़-अप तस्वीर लें या अपलोड करें।",
        "how_it_works_step2": "2. विवरण जोड़ें (वैकल्पिक): सटीकता बढ़ाने के लिए उम्र या शरीर का स्थान बताएं।",
        "how_it_works_step3": "3. परिणाम देखें: तुरंत संभावित स्थिति, विश्वास स्तर और अगला कदम जानें।",
        "try_example_title": "या कोई उदाहरण आज़माएं:",
        "example_mel": "उदाहरण 1: मेलेनोमा (गहरा त्वचा रंग)",
        "example_bcc": "उदाहरण 2: बेसल सेल (हल्का त्वचा रंग)",
        "example_nv": "उदाहरण 3: सामान्य तिल (मध्यम त्वचा रंग)",
        "example_badge": "🏷️ उदाहरण मामला",
        "user_photo_badge": "📸 आपकी फ़ोटो",
        "dismiss_banner": "✕ बंद करें",

        # Progress messages
        "progress_looking": "आपकी तस्वीरों की गुणवत्ता जांची जा रही है...",
        "progress_models": "5 डीप-लर्निंग मॉडलों द्वारा विश्लेषण किया जा रहा है...",
        "progress_preparing": "परिणाम संकलित कर अंतिम रिपोर्ट तैयार की जा रही है...",
        "toast_done": "✅ विश्लेषण सफलतापूर्वक पूर्ण हुआ!",

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
    },

    "kn": {
        # App identity
        "app_title": "ವಿಶ್ವಾಸ (VISHWAS): ಚರ್ಮ ಆರೋಗ್ಯ ತಪಾಸಣೆ",
        "app_subtitle_simple": "ಚರ್ಮದ ಬಣ್ಣ-ತಾರತಮ್ಯವಿಲ್ಲದ, ಸಮಾನ ಮತ್ತು ನಿಖರ ಚರ್ಮ ಆರೋಗ್ಯ ತಪಾಸಣೆಯ ಸಂಶೋಧನಾ ಮಾದರಿ.",
        "app_subtitle_expert": "ಪ್ರಮಾಣೀಕೃತ 5-ಫೋಲ್ಡ್ ಕ್ರಾಸ್-ವ್ಯಾಲಿಡೇಶನ್, ಮಲ್ಟಿಮೋಡಲ್ ಫ್ಯೂಷನ್ ಮತ್ತು ನ್ಯಾಯಸಮ್ಮತತೆಯ ಆಡಿಟಿಂಗ್ ವ್ಯವಸ್ಥೆ.",
        "mode_selector": "ಇಂಟರ್ಫೇಸ್ ಮೋಡ್",
        "mode_simple": "ಸರಳ (ಮಾರ್ಗದರ್ಶಿತ)",
        "mode_expert": "ತಜ್ಞ (ತಾಂತ್ರಿಕ)",
        "language_selector": "ಭಾಷೆ / Language / भाषा",
        "privacy_badge": "🔒 ಸಂಪೂರ್ಣ ಖಾಸಗಿ ಮತ್ತು ಸುರಕ್ಷಿತ: ನಿಮ್ಮ ಫೋಟೋಗಳನ್ನು ಮೆಮೊರಿಯಲ್ಲಿ ಮಾತ್ರ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತದೆ ಮತ್ತು ಸಂಗ್ರಹಿಸುವುದಿಲ್ಲ.",
        "medical_disclaimer": "⚠️ ಇದು ಸಂಶೋಧನೆ ಮತ್ತು ಶಿಕ್ಷಣದ ಮಾದರಿಯಾಗಿದೆ, ವೈದ್ಯಕೀಯ ಸಾಧನವಲ್ಲ. ವೈದ್ಯಕೀಯ ಸಲಹೆಗಾಗಿ ಅರ್ಹ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",

        # Theme
        "theme_selector": "ಥೀಮ್ / Mode",
        "theme_auto": "ಆಟೋ (ಸಿಸ್ಟಮ್ ಅನುಸಾರ)",
        "theme_light": "ಲೈಟ್ (ತಿಳಿ)",
        "theme_dark": "ಡಾರ್ಕ್ (ಗಾಢ)",

        # Navigation tabs
        "tab_check": "🔬 ಚರ್ಮದ ಮಚ್ಚೆಯ ತಪಾಸಣೆ",
        "tab_fairness": "⚖️ ಮಾದರಿ ಎಷ್ಟು ನ್ಯಾಯಸಮ್ಮತವಾಗಿದೆ?",
        "tab_reliability": "📈 ಮಾದರಿ ಎಷ್ಟು ವಿಶ್ವಾಸಾರ್ಹವಾಗಿದೆ?",
        "tab_explorer": "🗂️ ಡೇಟಾ ಎಕ್ಸ್‌ಪ್ಲೋರರ್",
        "tab_about": "📚 ಡೇಟಾಸೆಟ್ ವಿವರ",
        "tab_conditions_guide": "🩺 ಚರ್ಮ ರೋಗಗಳ ಕೈಪಿಡಿ",

        # Step 1: Photos
        "step1_title": "ಹಂತ 1: ಚರ್ಮದ ಮಚ್ಚೆಯ ಫೋಟೋಗಳನ್ನು ಸೇರಿಸಿ",
        "step1_desc": "ಮಚ್ಚೆ, ಗಂಟು ಅಥವಾ ಕಲೆಯ ಫೋಟೋಗಳನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ. ಹತ್ತಿರದ ಮತ್ತು ವರ್ಧಿತ (ಡರ್ಮೋಸ್ಕೋಪಿಕ್) ಎರಡೂ ಫೋಟೋಗಳನ್ನು ನೀಡುವುದು ಹೆಚ್ಚು ನಿಖರ ಫಲಿತಾಂಶ ನೀಡುತ್ತದೆ.",
        "label_closeup": "1. ಚರ್ಮದ ಮಚ್ಚೆಯ ಹತ್ತಿರದ (ಕ್ಲೋಸ್-ಅಪ್) ಫೋಟೋ",
        "help_closeup": "ಫೋನ್ ಕ್ಯಾಮೆರಾದಿಂದ ಸುಮಾರು 10-15 ಸೆಂ.ಮೀ ದೂರದಿಂದ ಮಚ್ಚೆಯ ಮೇಲೆ ಸ್ಪಷ್ಟವಾಗಿ ಫೋಕಸ್ ಮಾಡಿ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "label_dermo": "2. ವರ್ಧಿತ (ಡರ್ಮೋಸ್ಕೋಪ್) ಫೋಟೋ, ನಿಮ್ಮ ಬಳಿ ಇದ್ದರೆ",
        "help_dermo": "ಡರ್ಮಟೋಸ್ಕೋಪ್ ಲೆನ್ಸ್ ಅಥವಾ ಮ್ಯಾಕ್ರೋ ಲಗತ್ತಿನೊಂದಿಗೆ ತೆಗೆದ ವರ್ಧಿತ ಫೋಟೋ (ಐಚ್ಛಿಕ).",
        "take_photo_btn": "📸 ಕ್ಯಾಮೆರಾ ಬಳಸಿ ಫೋಟೋ ತೆಗೆಯಿರಿ",
        "camera_closeup": "ಹತ್ತಿರದ ಫೋಟೋ ಸೆರೆಹಿಡಿಯಿರಿ",
        "camera_dermo": "ವರ್ಧಿತ ಫೋಟೋ ಸೆರೆಹಿಡಿಯಿರಿ",
        "single_photo_notice": "ℹ️ ಸೂಚನೆ: ನೀವು ಒಂದೇ ಫೋಟೋ ನೀಡಿದ್ದೀರಿ. AI ವಿಶ್ಲೇಷಿಸುತ್ತದೆ, ಆದರೆ ಎರಡೂ ಫೋಟೋಗಳಿದ್ದರೆ ಫಲಿತಾಂಶ ಹೆಚ್ಚು ವಿಶ್ವಾಸಾರ್ಹವಾಗಿರುತ್ತದೆ.",
        "photo_tips_title": "💡 ಉತ್ತಮ ಫೋಟೋ ತೆಗೆಯಲು ಸಲಹೆಗಳು",
        "photo_tips_content": "• **ಉತ್ತಮ ಬೆಳಕು**: ನೈಸರ್ಗಿಕ ಹಗಲು ಬೆಳಕು ಅಥವಾ ಸ್ಪಷ್ಟ ಫ್ಲ್ಯಾಶ್ ಬಳಸಿ.\n• **ಸ್ಪಷ್ಟ ಫೋಕಸ್**: ಫೋಟೋ ಮಸುಕಾಗದಂತೆ ಪರದೆಯ ಮೇಲೆ ಮಚ್ಚೆಯನ್ನು ಸ್ಪರ್ಶಿಸಿ ಫೋಕಸ್ ಮಾಡಿ.\n• **ಫ್ರೇಮ್ ತುಂಬಿಸಿ**: ಮಚ್ಚೆ ಫ್ರೇಮ್‌ನ 30-50% ಜಾಗವನ್ನು ಆವರಿಸಿರಬೇಕು.\n• **ಫಿಲ್ಟರ್ ಬೇಡ**: ಚರ್ಮವನ್ನು ನಯಗೊಳಿಸುವ ಬ್ಯೂಟಿ ಫಿಲ್ಟರ್‌ಗಳನ್ನು ಆಫ್ ಮಾಡಿ.\n• **ಗಾತ್ರದ ಉಲ್ಲೇಖ**: ಸಾಧ್ಯವಾದರೆ ಮಚ್ಚೆಯ ಪಕ್ಕದಲ್ಲಿ ನಾಣ್ಯ ಅಥವಾ ಸಣ್ಣ ಅಳತೆಪಟ್ಟಿ ಇರಿಸಿ.",

        # Quality warnings
        "warn_blurry": "⚠️ ಈ ಫೋಟೋ ಮಸುಕಾಗಿದೆ. ಸ್ಪಷ್ಟ ಫೋಕಸ್‌ನಲ್ಲಿ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯುವುದರಿಂದ ಹೆಚ್ಚು ನಿಖರ ಫಲಿತಾಂಶ ದೊರೆಯುತ್ತದೆ.",
        "warn_too_dark": "⚠️ ಈ ಫೋಟೋ ಕತ್ತಲೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ.",
        "warn_too_bright": "⚠️ ಈ ಫೋಟೋ ಅತಿಯಾದ ಬೆಳಕು ಅಥವಾ ಪ್ರತಿಫಲನ ಹೊಂದಿದೆ. ದಯವಿಟ್ಟು ನೇರ ಪ್ರತಿಫಲನವಿಲ್ಲದೆ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.",
        "warn_low_res": "⚠️ ಫೋಟೋದ ರೆಸಲ್ಯೂಶನ್ ಕಡಿಮೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಹೆಚ್ಚಿನ ರೆಸಲ್ಯೂಶನ್ ಫೋಟೋ ಅಪ್ಲೋಡ್ ಮಾಡಿ.",

        # Step 2: Patient details
        "step2_title": "ಹಂತ 2: ರೋಗಿಯ ವಿವರಗಳು (ಐಚ್ಛಿಕ)",
        "step2_desc": "ಹೆಚ್ಚಿನ ವಿವರಗಳನ್ನು ನೀಡುವುದರಿಂದ ಫಲಿತಾಂಶದ ನಿಖರತೆ ಹೆಚ್ಚಬಹುದು. ಎಲ್ಲಾ ಕ್ಷೇತ್ರಗಳು ಸಂಪೂರ್ಣ ಐಚ್ಛಿಕ.",
        "label_age": "ಅಂದಾಜು ವಯಸ್ಸು (ವರ್ಷಗಳು)",
        "help_age": "ವಯಸ್ಸಿನ ವಿವರವು ವಯೋಸಹಜ ಮಚ್ಚೆಗಳು ಮತ್ತು ಗಂಭೀರ ಸ್ಥಿತಿಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಲು AI ಗೆ ನೆರವಾಗುತ್ತದೆ.",
        "dont_know_age": "ಖಚಿತ ವಯಸ್ಸು ತಿಳಿದಿಲ್ಲ",
        "label_sex": "ಹುಟ್ಟಿನ ಜೈವಿಕ ಲಿಂಗ",
        "help_sex": "ಕೆಲವು ಚರ್ಮದ ಸ್ಥಿತಿಗಳು ಜೈವಿಕ ಲಿಂಗಗಳಾದ್ಯಂತ ವಿಭಿನ್ನವಾಗಿ ಕಾಣಿಸಿಕೊಳ್ಳುತ್ತವೆ.",
        "sex_male": "ಪುರುಷ",
        "sex_female": "ಮಹಿಳೆ",
        "sex_unknown": "ಹೇಳಲು ಇಚ್ಛಿಸುವುದಿಲ್ಲ",
        "label_site": "ದೇಹದ ಯಾವ ಭಾಗದಲ್ಲಿ ಈ ಮಚ್ಚೆಯಿದೆ?",
        "help_site": "ದೇಹದ ಭಾಗದ ವಿವರವು ಚರ್ಮರೋಗದ ನಿಖರ ವಿಶ್ಲೇಷಣೆಗೆ ನಿರ್ಣಾಯಕ ಸಂದರ್ಭ ಒದಗಿಸುತ್ತದೆ.",
        "site_trunk": "🫁 ಎದೆ, ಬೆನ್ನು ಅಥವಾ ಹೊಟ್ಟೆ",
        "site_head_neck": "👤 ತಲೆ, ಮುಖ ಅಥವಾ ಕುತ್ತಿಗೆ",
        "site_lower_ext": "🦵 ಕಾಲು, ತೊಡೆ ಅಥವಾ ಸೊಂಟ",
        "site_upper_ext": "💪 ಕೈ ಅಥವಾ ಭುಜ",
        "site_hand": "✋ ಅಂಗೈ",
        "site_foot": "🦶 ಪಾದ",
        "site_genital": "🩲 ಜನನಾಂಗದ ಭಾಗ",
        "site_unknown": "❓ ಇತರ / ಖಚಿತವಿಲ್ಲ",

        # Step 3: Result & Pinned Strip
        "step3_title": "ಹಂತ 3: ಫಲಿತಾಂಶ ಮತ್ತು ಮುಂದಿನ ಕ್ರಮ",
        "result_heading": "AI ಕ್ಲಿನಿಕಲ್ ಮೌಲ್ಯಮಾಪನ",
        "most_likely": "ಹೆಚ್ಚಾಗಿ ಸಂಭವನೀಯ ಸ್ಥಿತಿ",
        "most_likely_uncertain": "ಹೆಚ್ಚಾಗಿ ಸಂಭವನೀಯ ಸ್ಥಿತಿ (ಅನಿಶ್ಚಿತ)",
        "confidence_level": "AI ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "conf_high": "ಹೆಚ್ಚಿನ ವಿಶ್ವಾಸ",
        "conf_med": "ಮಧ್ಯಮ ವಿಶ್ವಾಸ",
        "conf_low": "ಕಡಿಮೆ ವಿಶ್ವಾಸ (ವೈದ್ಯರ ತಪಾಸಣೆ ಅಗತ್ಯ)",
        "ai_estimate_tag": "ℹ️ ಇದು AI ಅಂದಾಜು, ವೈದ್ಯಕೀಯ ರೋಗನಿರ್ಣಯವಲ್ಲ.",
        "rerun_btn": "🔄 ಪುನಃ ವಿಶ್ಲೇಷಿಸಿ",
        "start_over_btn": "🔄 ಮೊದಲಿನಿಂದ ಪ್ರಾರಂಭಿಸಿ",
        "next_steps_title": "ನೀವು ಮುಂದೆ ಏನು ಮಾಡಬಹುದು",
        "advice_malignant": "**ದಯವಿಟ್ಟು ಶೀಘ್ರದಲ್ಲೇ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.** AI ಗಂಭೀರ ಲಕ್ಷಣಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಿದೆ, ಬಯಾಪ್ಸಿ ಅಥವಾ ವೈದ್ಯಕೀಯ ಪರೀಕ್ಷೆ ಅಗತ್ಯವಿರಬಹುದು.",
        "advice_premalignant": "**ದಯವಿಟ್ಟು ಶೀಘ್ರದಲ್ಲೇ ವೈದ್ಯರಿಗೆ ತೋರಿಸಿ.** ಈ ಮಚ್ಚೆ ಕ್ಯಾನ್ಸರ್-ಪೂರ್ವ ಲಕ್ಷಣಗಳನ್ನು ಹೊಂದಿದ್ದು, ಮುಂಚಿತ ಚಿಕಿತ್ಸೆ ಉತ್ತಮ.",
        "advice_benign": "**ಸಾಮಾನ್ಯವಾಗಿ ನಿರುಪದ್ರವಿ, ಆದರೆ ಬದಲಾವಣೆ ಕಂಡರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.** ನಿಯಮಿತ ತಪಾಸಣೆ ಮುಂದುವರಿಸಿ.",
        "advice_nontumour": "**ಚಿಕಿತ್ಸೆ ಬೇಕಾಗಬಹುದು, ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.** ಇದು ಗಡ್ಡೆಯ ಬದಲು ಉರಿಯೂತ ಅಥವಾ ರಕ್ತನಾಳದ ಸ್ಥಿತಿಯಂತೆ ಕಾಣುತ್ತದೆ.",
        "clinical_review_banner": "**ವೃತ್ತಿಪರ ವೈದ್ಯಕೀಯ ಪರಿಶೀಲನೆಯ ಶಿಫಾರಸು**: ಹೆಚ್ಚಿನ ಅಪಾಯದ ಲಕ್ಷಣಗಳು ಅಥವಾ ಮಾದರಿಯ ಕಡಿಮೆ ನಿಖರತೆಯಿಂದಾಗಿ, ವೈದ್ಯರಿಂದ ನೇರ ತಪಾಸಣೆ ಮಾಡಿಸಲು ಬಲವಾಗಿ ಶಿಫಾರಸು ಮಾಡುತ್ತೇವೆ.",
        "top3_title": "AI ಪರಿಗಣಿಸಿದ ಪ್ರಮುಖ 3 ಸಾಧ್ಯತೆಗಳು",
        "show_details": "🔬 ತಾಂತ್ರಿಕ ಮತ್ತು ಮಾದರಿ ವಿವರಗಳನ್ನು ನೋಡಿ",
        "download_pdf_btn": "📥 ಫಲಿತಾಂಶ ಸಾರಾಂಶ ಡೌನ್‌ಲೋಡ್ (PDF)",
        "download_json_btn": "📥 ಫಲಿತಾಂಶ ಡೌನ್‌ಲೋಡ್ (JSON)",
        "pdf_fallback_notice": "ℹ️ ಸೂಚನೆ: ಕ್ಲಿನಿಕಲ್ ಸ್ಪಷ್ಟತೆಗಾಗಿ PDF ವರದಿಯನ್ನು ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ರಚಿಸಲಾಗಿದೆ.",

        # Triage verdicts (without duplicate emoji)
        "triage_red_title": "ಕೂಡಲೇ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ",
        "triage_red_text": "AI ಗಂಭೀರ ಅಥವಾ ಕ್ಯಾನ್ಸರ್-ಪೂರ್ವ ಲಕ್ಷಣಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಿದೆ. ದಯವಿಟ್ಟು ಕೂಡಲೇ ಚರ್ಮರೋಗ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "triage_amber_title": "ತಪಾಸಣೆ ಮಾಡಿಸಿಕೊಳ್ಳಿ",
        "triage_amber_text": "AI ಉರಿಯೂತ ಅಥವಾ ಅಸಾಮಾನ್ಯ ಲಕ್ಷಣಗಳನ್ನು ಕಂಡಿದೆ, ಅಥವಾ ವಿಶ್ವಾಸ ಮಧ್ಯಮವಾಗಿದೆ. ವೈದ್ಯರ ತಪಾಸಣೆ ಸೂಕ್ತ.",
        "triage_green_title": "ಬಹುಶಃ ನಿರುಪದ್ರವಿ, ಗಮನವಿರಲಿ",
        "triage_green_text": "ಈ ಮಚ್ಚೆ ಸುರಕ್ಷಿತವಾಗಿ ಕಾಣುತ್ತದೆ ಮತ್ತು ಅಪಾಯ ಕಡಿಮೆ. ನಿಯಮಿತವಾಗಿ ಗಮನಿಸಿ, ಬದಲಾವಣೆ ಕಂಡರೆ ವೈದ್ಯರಿಗೆ ತೋರಿಸಿ.",
        "triage_grey_title": "ಸ್ಪಷ್ಟವಾಗಿಲ್ಲ, ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ",
        "triage_grey_text": "ಫೋಟೋದ ಗುಣಮಟ್ಟ ಅಥವಾ ಕಡಿಮೆ ವಿಶ್ವಾಸದಿಂದಾಗಿ ನಿಖರ ಅಂದಾಜು ಸಾಧ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟ ಬೆಳಕಿನಲ್ಲಿ ಮತ್ತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.",

        # Modal Popup & Pinned Strip
        "popup_title": "ಕ್ಲಿನಿಕಲ್ ಮೌಲ್ಯಮಾಪನ ಫಲಿತಾಂಶ",
        "popup_close_btn": "✕ ಮುಚ್ಚಿ",
        "view_details_btn": "🔍 ವಿವರಗಳು ಮತ್ತು ಫಲಿತಾಂಶ ನೋಡಿ",
        "result_strip_title": "ಇತ್ತೀಚಿನ ಮೌಲ್ಯಮಾಪನ ಸಾರಾಂಶ",
        "evidence_line": "ಪರೀಕ್ಷೆಯಲ್ಲಿ, ಮಾದರಿಯು ಇಷ್ಟು ವಿಶ್ವಾಸ ಹೊಂದಿದ್ದಾಗ, ಅದು ಸುಮಾರು {accuracy}% ಬಾರಿ ಸರಿಯಾಗಿತ್ತು.",

        # Empty state & examples
        "how_it_works_title": "👋 ಇದು ಹೇಗೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ (3 ಸುಲಭ ಹಂತಗಳು)",
        "how_it_works_step1": "1. ಫೋಟೋ ಸೇರಿಸಿ: ಚರ್ಮದ ಮಚ್ಚೆಯ ಸ್ಪಷ್ಟ ಕ್ಲೋಸ್-ಅಪ್ ಫೋಟೋ ತೆಗೆಯಿರಿ ಅಥವಾ ಅಪ್ಲೋಡ್ ಮಾಡಿ.",
        "how_it_works_step2": "2. ವಿವರ ಸೇರಿಸಿ (ಐಚ್ಛಿಕ): ನಿಖರತೆ ಹೆಚ್ಚಿಸಲು ವಯಸ್ಸು ಅಥವಾ ದೇಹದ ಭಾಗದ ವಿವರ ನೀಡಿ.",
        "how_it_works_step3": "3. ಫಲಿತಾಂಶ ನೋಡಿ: ತಕ್ಷಣವೇ ಸಂಭವನೀಯ ಸ್ಥಿತಿ, ವಿಶ್ವಾಸ ಮಟ್ಟ ಮತ್ತು ಮುಂದಿನ ಕ್ರಮ ತಿಳಿಯಿರಿ.",
        "try_example_title": "ಅಥವಾ ಉದಾಹರಣೆಯನ್ನು ಪ್ರಯತ್ನಿಸಿ:",
        "example_mel": "ಉದಾಹರಣೆ 1: ಮೆಲನೋಮ (ಗಾಢ ಚರ್ಮದ ಬಣ್ಣ)",
        "example_bcc": "ಉದಾಹರಣೆ 2: ಬೇಸಲ್ ಸೆಲ್ (ತಿಳಿ ಚರ್ಮದ ಬಣ್ಣ)",
        "example_nv": "ಉದಾಹರಣೆ 3: ಸಾಮಾನ್ಯ ಮಚ್ಚೆ (ಮಧ್ಯಮ ಚರ್ಮದ ಬಣ್ಣ)",
        "example_badge": "🏷️ ಉದಾಹರಣೆ ಪ್ರಕರಣ",
        "user_photo_badge": "📸 ನಿಮ್ಮ ಫೋಟೋ",
        "dismiss_banner": "✕ ಮುಚ್ಚಿ",

        # Progress messages
        "progress_looking": "ನಿಮ್ಮ ಫೋಟೋಗಳ ಗುಣಮಟ್ಟವನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ...",
        "progress_models": "5 ಆಳವಾದ ಕಲಿಕೆಯ ಮಾದರಿಗಳ ಮೂಲಕ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "progress_preparing": "ಫಲಿತಾಂಶಗಳನ್ನು ಒಗ್ಗೂಡಿಸಿ ಸಾರಾಂಶವನ್ನು ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ...",
        "toast_done": "✅ ವಿಶ್ಲೇಷಣೆ ಯಶಸ್ವಿಯಾಗಿ ಪೂರ್ಣಗೊಂಡಿದೆ!",

        # Glossary
        "glossary_title": "📖 ಸರಳ ಶಬ್ದಕೋಶ",
        "glossary_lesion": "**ಗಾಯ / ಕಲೆ (Lesion)**: ಚರ್ಮದ ಮೇಲಿನ ಯಾವುದೇ ಅಸಾಮಾನ್ಯ ಕಲೆ, ಮಚ್ಚೆ, ಗಂಟು ಅಥವಾ ದದ್ದು.",
        "glossary_dermoscopy": "**ಡರ್ಮೋಸ್ಕೊಪಿ (Dermoscopy)**: ಚರ್ಮರೋಗ ತಜ್ಞರು ಚರ್ಮದ ಆಳವನ್ನು ನೋಡಲು ಬಳಸುವ ವಿಶೇಷ ವರ್ಧಕ ಮಸೂರ.",
        "glossary_melanoma": "**ಮೆಲನೋಮ (Melanoma)**: ವರ್ಣದ್ರವ್ಯ ಕೋಶಗಳಿಂದ ಉಂಟಾಗುವ ಗಂಭೀರ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್. ಶೀಘ್ರ ಪತ್ತೆ ಅತ್ಯಂತ ಅಗತ್ಯ.",
        "glossary_bcc": "**ಬೇಸಲ್ ಸೆಲ್ ಕಾರ್ಸಿನೋಮ (BCC)**: ಸಾಮಾನ್ಯ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್; ಸಾಮಾನ್ಯವಾಗಿ ನಿಧಾನವಾಗಿ ಬೆಳೆಯುತ್ತದೆ ಮತ್ತು ಚಿಕಿತ್ಸೆಗೆ ಸ್ಪಂದಿಸುತ್ತದೆ.",
        "glossary_f1": "**ಮ್ಯಾಕ್ರೋ-F1 ಸ್ಕೋರ್**: ಅಪರೂಪದ ಮತ್ತು ಸಾಮಾನ್ಯ ರೋಗಗಳನ್ನು ಸಮಾನವಾಗಿ ಪರಿಗಣಿಸುವ ಸಮತೋಲಿತ ನಿಖರತೆಯ ಮಾನದಂಡ (0 ರಿಂದ 1).",
        "glossary_calibration": "**ಕ್ಯಾಲಿಬ್ರೇಶನ್ (Calibration)**: AI ಯ ಶೇಕಡಾವಾರು ವಿಶ್ವಾಸವು ನೈಜ ಪ್ರಪಂಚದ ವೈದ್ಯಕೀಯ ಖಚಿತತೆಗೆ ಎಷ್ಟು ಹೊಂದಿಕೆಯಾಗುತ್ತದೆ ಎಂಬ ಅಳತೆ.",
        "glossary_tone_group": "**ಸ್ಕಿನ್ ಟೋನ್ ಗುಂಪು**: ನ್ಯಾಯಸಮ್ಮತತೆಯನ್ನು ಪರೀಕ್ಷಿಸಲು ಚರ್ಮದ ಬಣ್ಣಗಳ ವರ್ಗೀಕರಣ (G1 = ತಿಳಿ, G2 = ಮಧ್ಯಮ, G3 = ಗಾಢ).",

        # Chart explainers
        "chart_explain_fairness": "💡 **ಈ ಚಾರ್ಟ್‌ನ ಅರ್ಥವೇನು?** ತಿಳಿ (G1), ಮಧ್ಯಮ (G2) ಮತ್ತು ಗಾಢ (G3) ಚರ್ಮದ ಬಣ್ಣಗಳ ಜನರ ನಡುವೆ AI ನಿಖರತೆಯನ್ನು ಈ ಚಾರ್ಟ್ ಹೋಲಿಸುತ್ತದೆ. ನ್ಯಾಯಯುತ ಮಾದರಿಯು ಎಲ್ಲಾ ಗುಂಪುಗಳಲ್ಲೂ ಸಮಾನ ನಿಖರತೆ ಸಾಧಿಸುತ್ತದೆ.",
        "chart_explain_calibration": "💡 **ಈ ಚಾರ್ಟ್‌ನ ಅರ್ಥವೇನು?** ಉತ್ತಮವಾಗಿ ಮಾಪನಾಂಕ ಮಾಡಿದ AI 90% ಸಮಯ ನಿಜವಾಗಿಯೂ ಸರಿಯಿದ್ದಾಗ ಮಾತ್ರ 90% ವಿಶ್ವಾಸ ವರದಿ ಮಾಡುತ್ತದೆ. ಚುಕ್ಕೆ ರೇಖೆಯ ಸಮೀಪವಿರುವುದು ಉತ್ತಮ ವಿಶ್ವಾಸಾರ್ಹತೆಯನ್ನು ತೋರಿಸುತ್ತದೆ.",
        "chart_explain_confusion": "💡 **ಈ ಚಾರ್ಟ್‌ನ ಅರ್ಥವೇನು?** ಈ ಮ್ಯಾಟ್ರಿಕ್ಸ್ AI ಯಾವ ರೋಗಗಳನ್ನು ಸರಿಯಾಗಿ ಗುರುತಿಸಿದೆ ಮತ್ತು ಯಾವ ರೋಗಗಳನ್ನು ಒಂದೇ ತರಹದ ಕಲೆಗಳೊಂದಿಗೆ ಗೊಂದಲಗೊಳಿಸಿದೆ ಎಂಬುದನ್ನು ತೋರಿಸುತ್ತದೆ.",
    }
}

# Detailed descriptions of the 11 classes with trilingual support (en, hi, kn)
CONDITION_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "AKIEC": {
        "en_name": "Actinic Keratosis / Bowen's Disease",
        "hi_name": "एक्टिनिक केराटोसिस / बोवेन रोग",
        "kn_name": "ಆಕ್ಟಿನಿಕ್ ಕೆರಾಟೋಸಿಸ್ / ಬೋವೆನ್ ಕಾಯಿಲೆ",
        "en_desc": "A rough, scaly patch on sun-damaged skin considered pre-cancerous, which can occasionally develop into squamous cell cancer if left untreated.",
        "hi_desc": "धूप से क्षतिग्रस्त त्वचा पर एक खुरदरा, पपड़ीदार धब्बा जो पूर्व-कैंसर माना जाता है और अनुपचारित रहने पर बढ़ सकता है।",
        "kn_desc": "ಬಿಸಿಲಿನಿಂದ ಹಾನಿಗೊಳಗಾದ ಚರ್ಮದ ಮೇಲಿನ ಒರಟಾದ, ಪೊರೆಯುಳ್ಳ ಕಲೆ. ಇದು ಕ್ಯಾನ್ಸರ್-ಪೂರ್ವ ಸ್ಥಿತಿಯಾಗಿದ್ದು, ಚಿಕಿತ್ಸೆ ನೀಡದಿದ್ದರೆ ಗಂಭೀರವಾಗಬಹುದು.",
        "category": "premalignant",
    },
    "BCC": {
        "en_name": "Basal Cell Carcinoma",
        "hi_name": "बेसल सेल कार्सिनोमा",
        "kn_name": "ಬೇಸಲ್ ಸೆಲ್ ಕಾರ್ಸಿನೋಮ (ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್)",
        "en_desc": "The most common form of skin cancer. It usually appears as a slightly shiny bump or pink patch, grows slowly, and rarely spreads if treated.",
        "hi_desc": "त्वचा कैंसर का सबसे आम प्रकार। यह अक्सर चमकदार उभार या गुलाबी धब्बे के रूप में दिखता है और उपचार से ठीक हो जाता है।",
        "kn_desc": "ಸಾಮಾನ್ಯ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್. ಇದು ಹೊಳೆಯುವ ಗುಳ್ಳೆ ಅಥವಾ ಗುಲಾಬಿ ಕಲೆಯಂತೆ ಕಾಣುತ್ತದೆ; ನಿಧಾನವಾಗಿ ಬೆಳೆಯುತ್ತದೆ ಮತ್ತು ಸಮಯಕ್ಕೆ ಚಿಕಿತ್ಸೆ ನೀಡಿದರೆ ಗುಣವಾಗುತ್ತದೆ.",
        "category": "malignant",
    },
    "BEN_OTH": {
        "en_name": "Other Benign Skin Lesion",
        "hi_name": "अन्य हानिरहित त्वचा उभार",
        "kn_name": "ಇತರ ನಿರುಪದ್ರವಿ ಚರ್ಮದ ಬೆಳವಣಿಗೆ",
        "en_desc": "A non-cancerous skin growth such as a skin tag, epidermoid cyst, or solar lentigo that presents no medical danger.",
        "hi_desc": "एक गैर-कैंसरयुक्त हानिरहित त्वचा वृद्धि जैसे मस्सा या सिस्ट जिससे कोई चिकित्सीय खतरा नहीं होता।",
        "kn_desc": "ಕ್ಯಾನ್ಸರ್ ರಹಿತವಾದ ನಿರುಪದ್ರವಿ ಚರ್ಮದ ಬೆಳವಣಿಗೆ ಅಥವಾ ಗಂಟು, ಇದರಿಂದ ಯಾವುದೇ ವೈದ್ಯಕೀಯ ಅಪಾಯವಿಲ್ಲ.",
        "category": "benign",
    },
    "BKL": {
        "en_name": "Benign Keratotic Lesion (Seborrheic Keratosis)",
        "hi_name": "सौम्य केराटोटिक घाव (सेबोरहाइक केराटोसिस)",
        "kn_name": "ಸೌಮ್ಯ ಕೆರಾಟೋಟಿಕ್ ಗಾಯ (ಸೆಬೊರ್ಹೆಕ್ ಕೆರಾಟೋಸಿಸ್)",
        "en_desc": "A very common, harmless warty or waxy spot that appears 'stuck onto' the skin as people age.",
        "hi_desc": "उम्र बढ़ने के साथ दिखने वाला एक बहुत ही सामान्य और हानिरहित मस्से जैसा धब्बा जो त्वचा पर चिपका हुआ सा लगता है।",
        "kn_desc": "ವಯಸ್ಸಾದಂತೆ ಸಾಮಾನ್ಯವಾಗಿ ಕಾಣಿಸಿಕೊಳ್ಳುವ ನಿರುಪದ್ರವಿ, ಮೇಣದಂಥ ಅಥವಾ ನರಲಿ ತರಹದ ಚರ್ಮದ ಕಲೆ.",
        "category": "benign",
    },
    "DF": {
        "en_name": "Dermatofibroma",
        "hi_name": "डर्माटोफ़ाइब्रोमा",
        "kn_name": "ಡರ್ಮಟೊಫೈಬ್ರೊಮಾ",
        "en_desc": "A firm, harmless small bump commonly found on the legs, often following an insect bite or minor skin injury.",
        "hi_desc": "आमतौर पर पैरों पर कीड़े के काटने या छोटी चोट के बाद बनने वाली एक सख्त, हानिरहित छोटी गांठ।",
        "kn_desc": "ಸಾಮಾನ್ಯವಾಗಿ ಕಾಲುಗಳ ಮೇಲೆ ಕೀಟ ಕಡಿತ ಅಥವಾ ಸಣ್ಣ ಗಾಯದ ನಂತರ ಉಂಟಾಗುವ ಗಟ್ಟಿಯಾದ, ಅಪಾಯವಿಲ್ಲದ ಸಣ್ಣ ಗಂಟು.",
        "category": "benign",
    },
    "INF": {
        "en_name": "Inflammatory or Infectious Condition",
        "hi_name": "सूजन या संक्रमण संबंधी स्थिति",
        "kn_name": "ಉರಿಯೂತ ಅಥವಾ ಸೋಂಕಿನ ಸ್ಥಿತಿ",
        "en_desc": "A reactive skin condition such as eczema, psoriasis, dermatitis, or a localized bacterial/fungal skin reaction.",
        "hi_desc": "एक्जिमा, सोरायसिस या संक्रमण जैसी सूजनयुक्त त्वचा स्थिति जिसके लिए मलहम या डॉक्टरी उपचार की आवश्यकता हो सकती है।",
        "kn_desc": "ಎಸ್ಜಿಮಾ, ಸೋರಿಯಾಸಿಸ್ ಅಥವಾ ಚರ್ಮದ ಸೋಂಕಿನಂತಹ ಉರಿಯೂತದ ಸ್ಥಿತಿ, ಇದಕ್ಕೆ ವೈದ್ಯಕೀಯ ಮುಲಾಮು ಅಥವಾ ಚಿಕಿತ್ಸೆ ಬೇಕಾಗಬಹುದು.",
        "category": "nontumour",
    },
    "MAL_OTH": {
        "en_name": "Other Rare Malignancy",
        "hi_name": "अन्य दुर्लभ त्वचा दुर्दमता",
        "kn_name": "ಇತರ ಅಪರೂಪದ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್",
        "en_desc": "A less common form of skin cancer (e.g. cutaneous lymphoma, Merkel cell carcinoma) requiring prompt histological examination.",
        "hi_desc": "त्वचा कैंसर का एक कम सामान्य रूप जिसके लिए तत्काल विशेषज्ञ जांच और बायोप्सी की आवश्यकता होती है।",
        "kn_desc": "ಅಪರೂಪದ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್ ರೂಪ, ಇದಕ್ಕೆ ತಕ್ಷಣ ತಜ್ಞರ ತಪಾಸಣೆ ಮತ್ತು ಬಯಾಪ್ಸಿ ಅಗತ್ಯವಿದೆ.",
        "category": "malignant",
    },
    "MEL": {
        "en_name": "Melanoma",
        "hi_name": "मेलेनोमा (गंभीर त्वचा कैंसर)",
        "kn_name": "ಮೆಲನೋಮ (ತೀವ್ರ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್)",
        "en_desc": "A high-risk type of skin cancer that begins in pigment-producing cells. Early detection and immediate excision are critical.",
        "hi_desc": "त्वचा कैंसर का एक गंभीर रूप। इसकी शीघ्र पहचान और विशेषज्ञ द्वारा त्वरित उपचार अत्यंत आवश्यक है।",
        "kn_desc": "ವರ್ಣದ್ರವ್ಯ ಕೋಶಗಳಿಂದ ಉಂಟಾಗುವ ಗಂಭೀರ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್. ಆರಂಭಿಕ ಹಂತದಲ್ಲೇ ಪತ್ತೆಹಚ್ಚುವುದು ಮತ್ತು ತಕ್ಷಣ ಚಿಕಿತ್ಸೆ ಪಡೆಯುವುದು ಅತ್ಯಗತ್ಯ.",
        "category": "malignant",
    },
    "NV": {
        "en_name": "Melanocytic Nevus (Normal Mole)",
        "hi_name": "मेलेनोसाइटिक नेवस (सामान्य तिल)",
        "kn_name": "ಮೆಲನೋಸೈಟಿಕ್ ನೆವಸ್ (ಸಾಮಾನ್ಯ ಮಚ್ಚೆ)",
        "en_desc": "A standard, benign collection of pigment cells. Normal moles are completely harmless but should be watched for changes.",
        "hi_desc": "वर्णक कोशिकाओं का एक सामान्य, हानिरहित तिल। यह पूरी तरह सुरक्षित होता है पर किसी भी बदलाव पर ध्यान देना चाहिए।",
        "kn_desc": "ವರ್ಣದ್ರವ್ಯ ಕೋಶಗಳ ಸಾಮಾನ್ಯ ಮತ್ತು ಸಂಪೂರ್ಣ ನಿರುಪದ್ರವಿ ಮಚ್ಚೆ. ಭವಿಷ್ಯದಲ್ಲಿ ಯಾವುದೇ ಬದಲಾವಣೆಗಳಿವೆಯೇ ಎಂದು ಗಮನಿಸುತ್ತಿರಬೇಕು.",
        "category": "benign",
    },
    "SCCKA": {
        "en_name": "Squamous Cell Carcinoma / Keratoacanthoma",
        "hi_name": "स्क्वैमस सेल कार्सिनोमा",
        "kn_name": "ಸ್ಕ್ವಾಮಸ್ ಸೆಲ್ ಕಾರ್ಸಿನೋಮ",
        "en_desc": "The second most common skin cancer, often appearing as a firm red nodule or scaly, crusting sore that does not heal.",
        "hi_desc": "दूसरा सबसे आम त्वचा कैंसर, जो अक्सर पपड़ीदार या न भरने वाले घाव के रूप में दिखता है।",
        "kn_desc": "ಎರಡನೇ ಅತ್ಯಂತ ಸಾಮಾನ್ಯ ಚರ್ಮದ ಕ್ಯಾನ್ಸರ್. ಸಾಮಾನ್ಯವಾಗಿ ಗುಣವಾಗದ ಕೆಂಪು ಗಂಟು ಅಥವಾ ಪೊರೆ ಕಲೆಯಾಗಿ ಕಾಣಿಸಿಕೊಳ್ಳುತ್ತದೆ.",
        "category": "malignant",
    },
    "VASC": {
        "en_name": "Vascular Lesion (Cherry Angioma, Pyogenic Granuloma)",
        "hi_name": "संवहनी घाव (रक्त वाहिका उभार)",
        "kn_name": "ರಕ್ತನಾಳದ ಗಂಟು (ವಾಸ್ಕುಲರ್ ಲೆಸಿಯನ್)",
        "en_desc": "A benign collection of small blood vessels in the skin, such as a bright red cherry angioma or vascular malformation.",
        "hi_desc": "रक्त वाहिकाओं का एक सामान्य, लाल रंग का हानिरहित उभार जिससे कोई चिकित्सीय खतरा नहीं होता।",
        "kn_desc": "ಚರ್ಮದಲ್ಲಿ ಸಣ್ಣ ರಕ್ತನಾಳಗಳ ನಿರುಪದ್ರವಿ ಕೆಂಪು ಬೆಳವಣಿಗೆ, ಇದರಿಂದ ಯಾವುದೇ ಕ್ಯಾನ್ಸರ್ ಅಪಾಯವಿಲ್ಲ.",
        "category": "nontumour",
    },
}


def t(key: str, lang: str = "en") -> str:
    """
    Returns the translated string for a key in the specified language.
    Fallback chain:
    1. Requested language dictionary (e.g. 'hi' or 'kn')
    2. English dictionary fallback ('en')
    3. Human-readable auto-formatted label (e.g. 'tab_conditions_guide' -> 'Conditions Guide')
       with a warning logged to console.
    Never exposes raw technical keys like 'tab_xxx' to the user.
    """
    if not key:
        return ""

    lang_dict = STRINGS.get(lang, STRINGS.get("en", {}))
    if key in lang_dict and lang_dict[key]:
        return lang_dict[key]

    en_dict = STRINGS.get("en", {})
    if key in en_dict and en_dict[key]:
        return en_dict[key]

    # Clean human-readable fallback
    clean = key
    for prefix in ("tab_", "label_", "btn_", "help_", "warn_", "advice_", "glossary_", "chart_explain_"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    readable = clean.replace("_", " ").strip().title()
    logging.warning("Missing translation key '%s' for lang '%s' (English fallback also missing). Returning '%s'", key, lang, readable)
    return readable
