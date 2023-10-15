def find_max_sentence_length(file_path):
    # Read the content of the specified file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into sentences using an empty line
    sentences = content.strip().split('\n\n')

    # Find the maximum sentence length in terms of tokens
    max_length = 0
    max_length_sentence = ''
    start_line_number = 0

    for line_number, sentence in enumerate(sentences, start=1):
        tokens = sentence.split('\n')
        if len(tokens) > max_length:
            max_length = len(tokens)
            max_length_sentence = sentence
            start_line_number = line_number

    print(f"The maximum sentence length in terms of tokens is: {max_length}")
    # print(f"The sentence with the maximum length starts from line {start_line_number}:")
    # print(max_length_sentence)

# Analyze 'train', 'testa', and 'testb' files
file_paths = ['CoNLL2003_dataset/eng.train', 'CoNLL2003_dataset/eng.testa', 'CoNLL2003_dataset/eng.testb']

for file_path in file_paths:
    print(f"Analyzing file: {file_path}")
    find_max_sentence_length(file_path)
    print("\n")
