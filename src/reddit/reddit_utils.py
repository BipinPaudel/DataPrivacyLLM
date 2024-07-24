

def type_to_str(pii_type: str) -> str:
    if pii_type == "income_level":
        pii_type_str = "yearly income"
    elif pii_type == "education":
        pii_type_str = "level of education"
    elif pii_type == "pobp":
        pii_type_str = "place of birth"
    elif pii_type == "location":
        pii_type_str = "current place of living"
    elif pii_type == "relationship_status":
        pii_type_str = "relationship status"
    elif pii_type == 'sex':
        pii_type_str = 'gender'
    else:
        pii_type_str = pii_type

    return pii_type_str

def type_to_options(pii_type: str) -> str:
    if pii_type == "income_level":
        pii_type_str = " Choose from these options: (low, middle, high, very high); low if <30k USD, middle if 30-60k USD, high if 60-150k USD, very high if >150k USD."
        # pii_type_str = " Choose from these options: No income, Low (<30k USD), Medium (30-60k USD), High (60-150k USD), Very High (>150k USD)."
    elif pii_type == "education":
        pii_type_str = " Choose from these options: No Highschool, In Highschool, HS Diploma, In College, College Degree, PhD."
    elif pii_type == "sex":
        pii_type_str = " Choose from these options: Male, Female."
    elif pii_type == "relationship_status":
        pii_type_str = (
            # " Choose from these options: No relation, In Relation, Married, Divorced."
            " Choose from these options: Widowed, Single, Married, Engaged, In a relationship, Divorced"
        )
    elif pii_type == "age":
        pii_type_str = " Use the age of the author when he wrote the comment if date is available."
    else:
        pii_type_str = ""

    return pii_type_str