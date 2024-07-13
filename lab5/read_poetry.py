def read_poetry(filename='poetryFromTang.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    poetries = []
    poetry = ''
    for l in lines:
        l = l.strip()
        if len(l) < 1 and len(poetry) > 0:
            poetries.append(poetry)
            poetry = ''
        else:
            poetry += l
    return poetries


def read_poetry_sentences(filename='poetryFromTang.txt'):
    poetries = read_poetry(filename)
    poetry_sentences = []
    for p in poetries:
        sentences = p.split('。')
        for s in sentences:
            if len(s) > 0:
                poetry_sentences.append(s + '。')
    return poetry_sentences


if __name__ == '__main__':
    poetry = read_poetry()
    print(len(poetry))

    poetry_sentences = read_poetry_sentences()
    print(len(poetry_sentences))
