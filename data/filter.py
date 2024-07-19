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

def get_unique_private_attribute(profiles, feature='income'):
    unique_values = set()
    for profile in profiles:
        private_key = list(profile.review_pii['synth'].keys())[0]
        unique_private_value = profile.review_pii['synth'][private_key]['estimate']
        assert unique_private_value is not None
        unique_values.add(unique_private_value)
    
    if feature == 'age':
        # Initial list of integers from 1 to 100
        initial_list = list(range(1, 101))

        # Generate the list with range strings
        range_list = []

        # Iterate over the initial list in steps of 10
        for i in range(0, 100, 10):
            start = i + 1
            end = i + 10
            range_list.append(f"{start}-{end}")

        return list(set(range_list))
    return list(unique_values)

def get_topics_for_features(profiles, feature='income'):
    return list(set([profile.concised_topics for profile in profiles]))
        