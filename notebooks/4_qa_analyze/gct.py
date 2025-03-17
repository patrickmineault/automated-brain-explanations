import numpy as np

from neuro.analyze_helper import abbrev_question

LOOSE_MATCHES = {
    'Does the sentence involve the mention of a specific object or item?',
    'Does the input include a comparison or metaphor?',
    'Does the sentence express a sense of belonging or connection to a place or community?',
    'Does the text describe a journey?',
}
QUESTIONS_GCT_DICT_S02 = {
    'Is time mentioned in the input?': ('Times', None),
    # ('time', 212),
    'Does the input contain a measurement?': ('measurements', 171),
    # ('Measurements', None),
    'Does the sentence mention a specific location?': ('locations', 368),
    # ('Location names', None),
    'Does the text describe a mode of communication?': ('communication', 299),

    # specific qa-targets
    'Is the sentence abstract rather than concrete?': ('abstract descriptions', 'qa'),
    'Does the sentence contain a cultural reference?': ('cultural references', 'qa'),
    'Does the sentence include dialogue?': ('dialogue', 'qa'),
    # ('Dialogue', None),
    'Is the input related to a specific industry or profession?': ('Professions and Personal Backgrounds', None),
    # ('industry or profession', 'qa'),
    'Does the sentence contain a negation?': ('negations', 'qa'),
    'Does the input contain a number?': ('Numbers', None),
    # ('numbers', 'qa'),
    "Does the sentence express the narrator's opinion or judgment about an event or character?": ('opinions or judgments', 'qa'),
    'Does the sentence describe a personal or social interaction that leads to a change or revelation?': ('personal or interactions interactions', 'qa'),
    'Does the sentence describe a personal reflection or thought?': ('personal reflections or thoughts', 'qa'),
    'Does the sentence involve an expression of personal values or beliefs?': ('personal values or beliefs', 'qa'),
    'Does the sentence describe a physical action?': ('physical actions', 'qa'),
    'Does the input involve planning or organizing?': ('planning or organizing', 'qa'),
    'Does the sentence contain a proper noun?': ('proper nouns', 'qa'),
    'Does the sentence describe a relationship between people?': ('relationships between people', 'qa'),
    # ('Relationships', None)
    'Does the sentence describe a sensory experience?': ('Body parts', None),
    # ('sensory experiences', 'qa'),
    # ('Descriptive elements of scenes or objects', None),
    'Does the sentence involve the mention of a specific object or item?': ('Descriptive elements of scenes or objects', None),
    # ('specific objects or items', 'qa'),
    'Does the sentence include technical or specialized terminology?': ('technical or specialized terminology', 'qa'),

    # extra non-targeted
    'Does the sentence involve a description of physical environment or setting?': ('Direction and location descriptions', None),
    # ('Descriptive elements of scenes or objects', None),
    # ('Scenes and settings', None),
    # ('physical setting', 'roi_rename'),
    'Does the sentence describe a visual experience or scene?': ('Direction and location descriptions', None),
    # ('Descriptive elements of scenes or objects', None),
    # ('Scenes and settings', None),
    'Does the sentence involve spatial reasoning?': ('Spatial positioning and directions', None),
    'Does the input include a comparison or metaphor?': ('abstract descriptions', 'qa'),
    'Does the sentence express a sense of belonging or connection to a place or community?': ('Relationships', None),
    # ('Positive Emotional Reactions', None),
    # ('Sexual and Romantic Interactions', None),
    # ('personal or interactions interactions', 'qa'),
    # ('relationships between people', 'qa'),
    'Does the sentence describe a specific sensation or feeling?': ('Body parts', None),
    # ('sensory experiences', 'qa'),
    'Does the text include a planning or decision-making process?': ('planning or organizing', 'qa'),
    'Does the sentence include a personal anecdote or story?': ('Dialogue', None),
    # ('personal or interactions interactions', 'qa'),
    'Does the sentence involve a discussion about personal or social values?': ('personal values or beliefs', 'qa'),
    'Does the text describe a journey?': ('Spatial positioning and directions', None),
    # ('Direction and location descriptions', None),
    'Does the sentence describe a physical sensation?': ('Body parts', None),
    'Does the sentence include a direct speech quotation?': ('dialogue', 'qa'),
    # ('Dialogue', None),
    'Is the sentence reflective, involving self-analysis or introspection?': ('personal reflections or thoughts', 'qa'),
    # ('Introspection', None),
    # ('Personal growth and reflection', None),
    'Does the input describe a specific texture or sensation?':  ('sensory experiences', 'qa'),
    # # ('Clothing and Physical Appearance', None),
    # ('Body parts', None),
}

# OPTIONS
# ('action or movement', np.int64(158))
# ('age', np.int64(109))
# ('age', np.int64(160))
# ('agreement and questioning', np.int64(458))
# ('body language', np.int64(99))
# ('communication', np.int64(280))
# ('conflict resolution', np.int64(173))
# ('family and relationships', np.int64(368))
# ('food and drinks', np.int64(466))
# ('locations', np.int64(342))
# ('love and joy', np.int64(337))
# ('movement or action', np.int64(152))
# ('negative experiences', np.int64(403))
# ('numbers', np.int64(9))
# ('numbers or measurements', np.int64(408))
# ('physical injury', np.int64(148))
# ('vomiting, sickness', np.int64(395))
# ('Clothing and Physical Appearance', None)
# ('Colors', None)
# ('Dialogue', None)
# ('Direction and location descriptions', None)
# ('Gruesome body imagery', None)
# ('Introspection', None)
# ('Measurements', None)
# ('Numbers', None)
# ('Relationships', None)
# ('Scenes and settings', None)
# ('Times', None)
# ('Travel and location names', None)
# ('Years', None)
# ('Body parts', None)
# ('Conversational transitions', None)
# ('Descriptive elements of scenes or objects', None)
# ('Dialogue and responses', None)
# ('Fear and Avoidance', None)
# ('Garbage, food, and household items', None)
# ('Location names', None)
# ('Negative Emotional Reactions', None)
# ('Positive Emotional Reactions', None)
# ('Professions and Personal Backgrounds', None)
# ('Recognition', None)
# ('Secretive Or Covert Actions', None)
# ('Self-reflection and growth', None)
# ('Sexual and Romantic Interactions', None)


QUESTIONS_GCT_DICT_S03 = {
    'Is time mentioned in the input?': ('Times', None),
    # 'Does the input contain a measurement?': ('Measurements', None),
    'Does the input contain a measurement?': ('numbers or measurements', np.int64(408)),
    # 'Does the sentence mention a specific location?': ('Location names', None),
    'Does the sentence mention a specific location?': ('locations', 342),
    # 'Does the sentence mention a specific location?': ('Travel and location names', None),
    'Does the text describe a mode of communication?': ('communication', np.int64(280)),
    'Does the sentence include dialogue?': ('Dialogue', None),
    'Is the input related to a specific industry or profession?': ('Professions and Personal Backgrounds', None),
    # 'Does the input contain a number?': ('Numbers', None),
    'Does the input contain a number?': ('numbers', np.int64(9)),
    'Does the sentence describe a physical action?': ('action or movement', np.int64(158)),
    # 'Does the sentence describe a relationship between people?': ('Sexual and Romantic Interactions', None),
    'Does the sentence describe a relationship between people?': ('family and relationships', np.int64(368)),
    'Does the sentence involve a description of physical environment or setting?': ('Scenes and settings', None),
    'Does the sentence include a direct speech quotation?': ('Dialogue', None),
    'Is the sentence reflective, involving self-analysis or introspection?': ('Introspection', None),
    'Does the input describe a specific texture or sensation?': ('Body parts', None),
    'Does the sentence describe a visual experience or scene?': ('Descriptive elements of scenes or objects', None),
    'Does the sentence involve spatial reasoning?': ('Direction and location descriptions', None),
    #     'Does the input include a comparison or metaphor?': None,
    'Does the sentence express a sense of belonging or connection to a place or community?': ('family and relationships', np.int64(368)),
    #     'Does the sentence describe a specific sensation or feeling?': None,
    #     'Does the text include a planning or decision-making process?': None,
    #     'Does the sentence include a personal anecdote or story?': None,
    #     'Does the sentence involve a discussion about personal or social values?': None,
    'Does the text describe a journey?': ('Travel and location names', None),
    #     'Does the sentence describe a physical sensation?': None,
    #     'Does the sentence describe a sensory experience?': None,
    'Does the sentence involve the mention of a specific object or item?': ('Descriptive elements of scenes or objects', None),
    #     'Does the sentence include technical or specialized terminology?': None,
    #     'Does the input involve planning or organizing?': None,
    #     'Does the sentence contain a proper noun?': None,
    #     "Does the sentence express the narrator's opinion or judgment about an event or character?": None,
    # 'Does the sentence describe a personal or social interaction that leads to a change or revelation?': None,
    'Does the sentence describe a personal reflection or thought?': ('Self-reflection and growth', None),
    #     'Does the sentence involve an expression of personal values or beliefs?': None,
    # 'Does the sentence contain a negation?': ('negative experiences', np.int64(403)),
    #     'Is the sentence abstract rather than concrete?': None,
    #     'Does the sentence contain a cultural reference?': None,
}


def abbrev_question_to_original(q, subject='S02'):
    q = q.replace('*', '')
    for k, v in QUESTIONS_GCT_DICT_S02.items():
        if abbrev_question(k) == q:
            return k
    return None


def get_matched_lists(subject='S02'):
    if subject == 'S02':
        QUESTIONS_GCT_DICT = QUESTIONS_GCT_DICT_S02
    elif subject == 'S03':
        QUESTIONS_GCT_DICT = QUESTIONS_GCT_DICT_S03
    qa_list = list(QUESTIONS_GCT_DICT.keys())
    gemv_list = list(QUESTIONS_GCT_DICT.values())
    return qa_list, gemv_list


QS_35_SORTED_BEST_TO_WORST = [
    'Is the input related to a specific industry or profession?',
    'Does the input contain a measurement?',
    'Does the sentence mention a specific location?',
    'Does the sentence include a direct speech quotation?',
    'Does the input contain a number?',
    'Does the sentence include dialogue?',
    'Does the sentence describe a specific sensation or feeling?',
    'Does the text describe a journey?',
    'Does the text describe a mode of communication?',
    'Does the sentence contain a cultural reference?',
    'Does the sentence describe a relationship between people?',
    'Does the sentence involve the mention of a specific object or item?',
    'Does the sentence describe a physical action?',
    'Does the sentence describe a personal or social interaction that leads to a change or revelation?',
    'Is time mentioned in the input?',
    'Does the sentence contain a proper noun?',
    'Does the sentence involve an expression of personal values or beliefs?',
    'Does the sentence describe a physical sensation?',
    'Does the sentence involve a discussion about personal or social values?',
    'Does the sentence include a personal anecdote or story?',
    'Does the sentence include technical or specialized terminology?',
    'Does the sentence describe a visual experience or scene?',
    'Does the input involve planning or organizing?',
    'Does the sentence involve a description of physical environment or setting?',
    "Does the sentence express the narrator's opinion or judgment about an event or character?",
    'Does the sentence involve spatial reasoning?',
    'Does the sentence describe a personal reflection or thought?',
    'Does the sentence describe a sensory experience?',
    'Is the sentence reflective, involving self-analysis or introspection?',
    'Does the sentence contain a negation?',
    'Does the input describe a specific texture or sensation?',
    'Does the text include a planning or decision-making process?',
    'Does the sentence express a sense of belonging or connection to a place or community?',
    'Does the input include a comparison or metaphor?',
    'Is the sentence abstract rather than concrete?',
]


if __name__ == '__main__':
    for k in LOOSE_MATCHES:
        print(k, QUESTIONS_GCT_DICT[k])
