# Reverse mappings for attributes
ATTRIBUTES_REVERSE_MAPPINGS = {
    "insurance_type": {-1: "Unknown", 1: 'Medicare', 2: 'Medicaid', 3: 'Private', 4: 'Other'},
    "anesthesia": {-1: "Unknown", 0: 'general', 1: 'regional', 2: 'hybrid', 3: 'MAC'},
    "sex": {-1: "Unknown", 0: 'Female', 1: 'Male'},
    "ethnic_group": {-1: "Unknown or declined to state", 0: 'Non-Hispanic/Non-Latino', 1: 'Hispanic/Latino'},
    "race": {-1: "Unknown or declined to state", 0: 'White', 1: 'Other', 2: 'Asian', 3: 'Black', 4: 'Hawaiian or Pacific Islander', 5: 'Native American'},
    "race_ethnicity": {-1: "Unknown or declined to state", 0: 'White', 1: 'Asian', 2: 'Hispanic/Latino', 3: 'Non-Hispanic Other', 4: 'Black', 5: 'Hawaiian or Pacific Islander', 6: 'Native American'}
}

# Feature sets for fairness analysis
COMORBIDITY_FEATURES = ['Myocardial_Infarction', 'Congestive_Heart_Failure',
       'Peripheral_Vascular_Disease', 'Cerebrovascular_Disease', 'Dementia',
       'Chronic_Pulmonary_Disease', 'Rheumatic_Disease',
       'Peptic_Ulcer_Disease', 'Mild_Liver_Disease',
       'Diabetes_without_Chronic_Complication',
       'Diabetes_with_Chronic_Complication', 'Hemiplegia_or_Paraplegia',
       'Renal_Disease', 'Any_Malignancy', 'Moderate_or_Severe_Liver_Disease',
       'Metastatic_Solid_Tumor', 'AIDS_HIV']

IF_FEATURES = ['age', 'bmi', 'opioid_naive', 'Charlson_score', 'pre_pain_score',
       'mental_substance_abuse', 'mental_mood_disorders',
       'mental_anxiety_disorders', 'mental_behavioral_syndromes',
       'anesthesia_type_MAC', 'anesthesia_type_general',
       'anesthesia_type_hybrid', 'anesthesia_type_regional',
       'surg_family_Appendectomy', 'surg_family_CABG',
       'surg_family_ColorecResect', 'surg_family_ExciLysisPeriAdhesions',
       'surg_family_HysterecAbVag', 'surg_family_InguinHerniaRepair',
       'surg_family_KneeReplacement', 'surg_family_OophorectomyUniBi',
       'surg_family_OtherHand', 'surg_family_PartialExcBone',
       'surg_family_SpinalFusion', 'surg_family_TreatFracDisHipFemur',
       'surg_family_TreatFracDisLowExtremity', 'surg_family_cholecystectomy',
       'surg_family_laminectomy', 'surg_family_mastectomy',
       'surg_family_prostatectomy', 'surg_family_thoracotomy',
       'Myocardial_Infarction', 'Congestive_Heart_Failure',
       'Peripheral_Vascular_Disease', 'Cerebrovascular_Disease', 'Dementia',
       'Chronic_Pulmonary_Disease', 'Rheumatic_Disease',
       'Peptic_Ulcer_Disease', 'Mild_Liver_Disease',
       'Diabetes_without_Chronic_Complication',
       'Diabetes_with_Chronic_Complication', 'Hemiplegia_or_Paraplegia',
       'Renal_Disease', 'Any_Malignancy', 'Moderate_or_Severe_Liver_Disease',
       'Metastatic_Solid_Tumor', 'AIDS_HIV']

CLINICAL_FEATURES = ['age', 'Charlson_score', 'pre_pain_score', 'bmi']