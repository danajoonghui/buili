from sqlalchemy import select

from services.api.buili.database import SessionLocal, init_db
from services.api.buili.models import SpecChunk
from services.api.buili.pipeline import cosine_search


def main() -> None:
    init_db()
    with SessionLocal() as session:
        chunks = session.scalars(select(SpecChunk)).all()
        for result in cosine_search("outlet count north wall", list(chunks), top_k=5):
            print(result)


if __name__ == "__main__":
    main()

