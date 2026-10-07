from __future__ import annotations

POSITIVE_IAT_WORDS = (
    "Love",
    "Cheer",
    "Friend",
    "Pleasure",
    "Adore",
    "Cheerful",
    "Friendship",
    "Joyful",
    "Smiling",
    "Cherish",
    "Excellent",
    "Glad",
    "Joyous",
    "Spectacular",
    "Appealing",
    "Delight",
    "Excitement",
    "Laughing",
    "Attractive",
    "Fabulous",
    "Glorious",
    "Pleasing",
    "Beautiful",
    "Fantastic",
    "Happy",
    "Lovely",
    "Terrific",
    "Celebrate",
    "Enjoy",
    "Magnificent",
)

NEGATIVE_IAT_WORDS = (
    "Abuse",
    "Grief",
    "Poison",
    "Sadness",
    "Pain",
    "Despise",
    "Failure",
    "Nasty",
    "Angry",
    "Detest",
    "Horrible",
    "Negative",
    "Ugly",
    "Dirty",
    "Gross",
    "Evil",
    "Rotten",
    "Annoy",
    "Disaster",
    "Horrific",
    "Scorn",
    "Awful",
    "Disgust",
    "Hate",
    "Humiliate",
    "Selfish",
    "Tragic",
    "Bothersome",
    "Hatred",
    "Hurtful",
    "Sick",
)

IAT_WORDS = POSITIVE_IAT_WORDS + NEGATIVE_IAT_WORDS
IAT_WORD_LOOKUP = {word.lower(): word for word in IAT_WORDS}
NEGATIVE_IAT_SET = {word.lower() for word in NEGATIVE_IAT_WORDS}

# These terms are forbidden in chosen responses. Race and gender metadata are
# available to the sampler but must not be verbalized or used by the judge.
# `black`, `white`, and `skin` are interpreted contextually by identity_hits so
# legitimate clothing, background, hair, and anatomical descriptions survive.
IDENTITY_TERMS = {
    "african",
    "african-american",
    "asian",
    "black",
    "caucasian",
    "ethnicity",
    "female",
    "gender",
    "girl",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "hispanic",
    "indian",
    "latina",
    "latino",
    "male",
    "man",
    "nationality",
    "race",
    "racial",
    "she",
    "skin",
    "white",
    "woman",
    "boy",
}

PROMPT_METADATA_TERMS = IDENTITY_TERMS | {
    "demographic",
    "marginalized",
    "minority",
    "protected class",
}

TASK_IAT = "iat_3word"
TASK_PROFILE = "grounded_profile"
TASK_TYPES = {TASK_IAT, TASK_PROFILE}

IAT_OBJECTIVES = {
    "negative_count_reduction",
    "unsupported_negative_substitution",
}

PROFILE_OBJECTIVES = {
    "person_negativity_reduction",
    "unsupported_trait_removal",
    "identity_removal",
    "invented_biography_removal",
    "grounding_improvement",
}

EXPRESSION_CLASSES = {"negative", "neutral", "positive"}
TARGET_RACES = {"Black", "White"}
TARGET_GENDERS = {"Female", "Male"}

EXPRESSION_PAIR_AUDIT_VERSION = "august3-pairwise-expression-v2-contract"
EXPRESSION_PAIR_MINIMUM_CONFIDENCE = 0.70
EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE = 1
