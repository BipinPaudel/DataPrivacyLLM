def filter_profiles(profiles, feature, hardness=[1,2,3,4,5]):
    filtered_profiles = []
    for profile in profiles:
        pii_type_hardness_dict = pii_type_with_hardness_of_a_profile(profile)
        if feature in pii_type_hardness_dict and pii_type_hardness_dict[feature] in hardness:
            filtered_profiles.append(profile)
    return filtered_profiles

def pii_type_with_hardness_of_a_profile(profile):
    pii_type_hardness_dict = {}
    for pii_type, pii_desc in profile.review_pii['synth'].items():
        pii_type_hardness_dict[pii_type] = pii_desc.get('hardness')
    return pii_type_hardness_dict
        
        