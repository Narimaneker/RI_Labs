def parser_medline(filepath: str) -> dict[int, str]:
    docs = {}
    current_id = None
    current_text = []
    in_text = False

    with open(filepath, 'r') as f:
        for line in f:
            line =line.strip()
            if line.startswith(".I"):
                if current_id is not None:
                    docs[current_id] = ' '.join(current_text)
                current_id = int(line.split()[1])
                current_text = []
                in_text = False
            elif line.startswith(".W"):
                in_text = True
            elif in_text and line:
                current_text.append(line)
        # save the last doc 
        if current_id is not None:
            docs[current_id] = ' '.join(current_text)
    return docs