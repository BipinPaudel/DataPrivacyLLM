import re


def extract_from_brackets(text):
    # Use a regular expression to find content inside square brackets
    match = re.search(r'\[([^\]]+)\]', text)
    if match:
        return match.group(1).strip()
    return ''

def evaluate(profile, res):
    target = list(profile.review_pii['synth'].values())[0].get('estimate')
    output = extract_from_brackets(res)
    return target.lower()==output.lower(), output.lower()

def evaluate_age(profile, res):
    target = list(profile.review_pii['synth'].values())[0].get('estimate')
    output = extract_from_brackets(res)
    try:
        op = int(output)
        return (op - 5) <= target <= (op + 5),op
    except ValueError:
        return -1, -1
    
    