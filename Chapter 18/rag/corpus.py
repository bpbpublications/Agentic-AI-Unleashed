# rag/corpus.py
# ============================================================
# Curated Clinical Document Corpus
# Agentic AI Unleashed — Appendix Demo
#
# All content is drawn from public domain / openly licensed
# government and international health organization sources:
#
#   - USPSTF: U.S. Preventive Services Task Force
#             https://www.uspreventiveservicestaskforce.org
#             (U.S. Government work, public domain)
#
#   - AHA/ACC & ADA: Excerpts paraphrased from publicly
#              available guideline summaries
#
#   - FDA DailyMed: https://dailymed.nlm.nih.gov
#                   (U.S. Government work, public domain)
#
# Each document chunk is short and self-contained so that
# retrieval remains crisp and explainable in the demo.
# ============================================================

CLINICAL_DOCUMENTS: list[dict] = [

    # --------------------------------------------------------
    # HYPERTENSION
    # --------------------------------------------------------
    {
        "id": "uspstf-htn-screening",
        "title": "USPSTF: Hypertension Screening in Adults",
        "source": "U.S. Preventive Services Task Force, 2021",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/hypertension-in-adults-screening",
        "content": (
            "The USPSTF recommends screening for hypertension in adults 18 years or older. "
            "Screening should be conducted with blood pressure measurement. Adults aged 40 years "
            "or older, and younger adults at increased risk, should be screened annually. "
            "Adults aged 18 to 39 years with normal blood pressure (less than 130/85 mmHg) "
            "and without risk factors should be rescreened every 3 to 5 years. "
            "Hypertension is defined as a sustained systolic blood pressure of 130 mmHg or higher "
            "or a diastolic blood pressure of 80 mmHg or higher. "
            "This recommendation received a Grade A from the USPSTF, indicating high certainty "
            "of substantial net benefit."
        ),
    },
    {
        "id": "acc-aha-htn-treatment",
        "title": "ACC/AHA Hypertension Guideline: Treatment Thresholds",
        "source": "American College of Cardiology / American Heart Association, 2017 Summary",
        "url": "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065",
        "content": (
            "The 2017 ACC/AHA guideline recommends initiating antihypertensive drug treatment "
            "for adults with confirmed hypertension and known cardiovascular disease (CVD) or "
            "a 10-year ASCVD event risk of 10% or higher, at a blood pressure of 130/80 mmHg or higher. "
            "For adults with confirmed hypertension without additional markers of increased CVD risk, "
            "drug treatment is recommended at a blood pressure of 140/90 mmHg or higher. "
            "First-line agents include thiazide diuretics, calcium channel blockers (CCBs), "
            "ACE inhibitors (ACEIs), and angiotensin receptor blockers (ARBs). "
            "Lifestyle modifications — including weight loss, DASH diet, sodium reduction, "
            "increased physical activity, and moderation of alcohol — are recommended for all patients."
        ),
    },

    # --------------------------------------------------------
    # DIABETES SCREENING & MANAGEMENT
    # --------------------------------------------------------
    {
        "id": "uspstf-diabetes-screening",
        "title": "USPSTF: Prediabetes and Type 2 Diabetes Screening",
        "source": "U.S. Preventive Services Task Force, 2021",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/screening-for-prediabetes-and-type-2-diabetes",
        "content": (
            "The USPSTF recommends screening for prediabetes and type 2 diabetes in adults "
            "aged 35 to 70 years who have overweight or obesity (Grade B recommendation). "
            "Clinicians should offer or refer patients with prediabetes to effective preventive "
            "interventions. The fasting plasma glucose test, the 75-g oral glucose tolerance test, "
            "and the hemoglobin A1c (HbA1c) test are each acceptable methods for screening. "
            "Prediabetes is defined as a fasting glucose of 100-125 mg/dL, a 2-hour glucose of "
            "140-199 mg/dL on the OGTT, or an HbA1c of 5.7%-6.4%. "
            "Type 2 diabetes is diagnosed at fasting glucose >= 126 mg/dL, 2-hour glucose >= 200 mg/dL, "
            "or HbA1c >= 6.5%."
        ),
    },
    {
        "id": "ada-diabetes-glycemic-targets",
        "title": "ADA Standards of Care: Glycemic Targets in Type 2 Diabetes",
        "source": "American Diabetes Association Standards of Medical Care, 2024 Summary",
        "url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
        "content": (
            "For most nonpregnant adults with type 2 diabetes, the American Diabetes Association "
            "recommends an HbA1c target of less than 7.0%. Less stringent targets (less than 8.0%) "
            "may be appropriate for patients with a history of severe hypoglycemia, limited life "
            "expectancy, advanced complications, or where the burdens of treatment outweigh benefits. "
            "More stringent targets (less than 6.5%) may be considered for younger patients with "
            "short disease duration if achievable without significant hypoglycemia. "
            "Metformin remains the preferred initial pharmacologic agent for type 2 diabetes "
            "when tolerated and not contraindicated. GLP-1 receptor agonists and SGLT-2 inhibitors "
            "are recommended for patients with established CVD, heart failure, or chronic kidney disease."
        ),
    },

    # --------------------------------------------------------
    # STATIN THERAPY / CARDIOVASCULAR PREVENTION
    # --------------------------------------------------------
    {
        "id": "uspstf-statin-cvd",
        "title": "USPSTF: Statin Use for Primary Prevention of CVD in Adults",
        "source": "U.S. Preventive Services Task Force, 2022",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/statin-use-in-adults-preventive-medication",
        "content": (
            "The USPSTF recommends prescribing a statin for the primary prevention of CVD events "
            "and mortality for adults aged 40 to 75 years who have one or more CVD risk factors "
            "(dyslipidemia, diabetes, hypertension, or smoking) and an estimated 10-year CVD event "
            "risk of 10% or greater (Grade B). "
            "The USPSTF recommends selectively offering a statin for adults aged 40 to 75 with "
            "CVD risk factors and an estimated 10-year risk of 7.5% to less than 10% (Grade C). "
            "The 10-year CVD risk should be calculated using the Pooled Cohort Equations."
        ),
    },

    # --------------------------------------------------------
    # DRUG: METFORMIN
    # --------------------------------------------------------
    {
        "id": "fda-metformin-label",
        "title": "FDA DailyMed: Metformin Hydrochloride — Drug Label Summary",
        "source": "FDA DailyMed, National Library of Medicine",
        "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?query=metformin",
        "content": (
            "Metformin hydrochloride is a biguanide antihyperglycemic agent indicated as an adjunct "
            "to diet and exercise to improve glycemic control in adults and pediatric patients "
            "aged 10 years and older with type 2 diabetes mellitus. "
            "Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73 m2. "
            "The FDA recommends obtaining eGFR before initiating treatment and periodically thereafter. "
            "Metformin should be withheld before iodinated contrast procedures in patients with "
            "eGFR 30-60 mL/min/1.73 m2. "
            "The most common adverse effects are gastrointestinal (nausea, vomiting, diarrhea). "
            "Lactic acidosis is a rare but serious complication; risk is increased in patients with "
            "renal impairment, hepatic impairment, congestive heart failure, or excessive alcohol use. "
            "Metformin does not cause hypoglycemia when used as monotherapy."
        ),
    },

    # --------------------------------------------------------
    # DRUG: LISINOPRIL
    # --------------------------------------------------------
    {
        "id": "fda-lisinopril-label",
        "title": "FDA DailyMed: Lisinopril — Drug Label Summary",
        "source": "FDA DailyMed, National Library of Medicine",
        "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?query=lisinopril",
        "content": (
            "Lisinopril is an angiotensin-converting enzyme (ACE) inhibitor indicated for the "
            "treatment of hypertension, heart failure, and to improve survival in hemodynamically "
            "stable patients within 24 hours of acute myocardial infarction. "
            "Lisinopril is contraindicated in patients with a history of ACE inhibitor-associated "
            "angioedema and carries a BLACK BOX WARNING for fetal toxicity — discontinue as soon "
            "as pregnancy is detected. "
            "Key drug interactions: NSAIDs may reduce antihypertensive effect and worsen renal "
            "function; potassium-sparing diuretics increase risk of hyperkalemia; lithium levels "
            "may increase. Common adverse effects include dry cough (10-15% of patients), "
            "dizziness, and headache."
        ),
    },

    # --------------------------------------------------------
    # DRUG: ATORVASTATIN
    # --------------------------------------------------------
    {
        "id": "fda-atorvastatin-label",
        "title": "FDA DailyMed: Atorvastatin Calcium — Drug Label Summary",
        "source": "FDA DailyMed, National Library of Medicine",
        "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?query=atorvastatin",
        "content": (
            "Atorvastatin calcium is an HMG-CoA reductase inhibitor (statin) indicated to reduce "
            "elevated total cholesterol, LDL-C, and triglycerides, and to reduce the risk of MI "
            "and stroke in patients with multiple cardiovascular risk factors. "
            "Atorvastatin is contraindicated in patients with active liver disease and in pregnant "
            "or breastfeeding women. "
            "Myopathy and rhabdomyolysis are serious risks; risk is increased with high doses, "
            "hypothyroidism, renal impairment, and concurrent use of cyclosporine, niacin, "
            "fibrates, erythromycin, and azole antifungals. "
            "Patients should promptly report unexplained muscle pain, tenderness, or weakness."
        ),
    },

    # --------------------------------------------------------
    # COLORECTAL CANCER SCREENING
    # --------------------------------------------------------
    {
        "id": "uspstf-colorectal-screening",
        "title": "USPSTF: Colorectal Cancer Screening",
        "source": "U.S. Preventive Services Task Force, 2021",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/colorectal-cancer-screening",
        "content": (
            "The USPSTF recommends screening for colorectal cancer in all adults aged 45 to 75 years "
            "(Grade A for ages 50-75, Grade B for ages 45-49). "
            "Acceptable screening strategies include: annual high-sensitivity fecal immunochemical "
            "test (FIT); stool DNA-FIT every 1 to 3 years; CT colonography every 5 years; "
            "flexible sigmoidoscopy every 5 years; or colonoscopy every 10 years. "
            "Adults with a first-degree relative with colorectal cancer diagnosed before age 60 "
            "should begin screening at age 40 or 10 years before the relative's diagnosis."
        ),
    },

    # --------------------------------------------------------
    # ASPIRIN / PRIMARY CVD PREVENTION
    # --------------------------------------------------------
    {
        "id": "uspstf-aspirin-cvd",
        "title": "USPSTF: Aspirin Use to Prevent CVD",
        "source": "U.S. Preventive Services Task Force, 2022",
        "url": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/aspirin-to-prevent-cardiovascular-disease-and-cancer",
        "content": (
            "The USPSTF recommends against initiating low-dose aspirin use for the primary "
            "prevention of CVD in adults aged 60 years or older (Grade D recommendation). "
            "For adults aged 40 to 59 years who have a 10% or greater 10-year CVD risk, "
            "the decision to use low-dose aspirin should be individualized (Grade C). "
            "The potential for harm — primarily gastrointestinal bleeding and hemorrhagic stroke — "
            "increases with age and outweighs the benefit in older adults for primary prevention. "
            "These recommendations apply to primary prevention only and do not address aspirin "
            "for secondary prevention in patients with established CVD."
        ),
    },
]
