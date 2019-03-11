import pprint

import arxiv
import data


def get_candidates():
    with open('arxiv-input-data', 'r') as f:
        return f.readlines()


def process_all():
    arxiv_tags_in_db = set(i.arxiv_id for i in data.DB)
    candidates = get_candidates()
    resu = []

    for candidate_id in candidates:
        candidate_id = candidate_id.strip()
        if candidate_id not in arxiv_tags_in_db:
            # try adding
            answer = arxiv.query(id_list=[candidate_id])[0]

            resu.append(data.Paper(
                title=answer['title'],
                desc=answer['summary'],
                authors=answer['authors'],
                arxiv_id=candidate_id
            ))

    pprint.pprint(resu)


if __name__ == '__main__':
    process_all()
