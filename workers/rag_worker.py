from sqlalchemy import select

from services.api.buili.database import SessionLocal, init_db
from services.api.buili.gpu import force_gpu_7
from services.api.buili.models import SpecChunk
from services.api.buili.pipeline import rag_answer

force_gpu_7()


def main() -> None:
    init_db()
    with SessionLocal() as session:
        chunks = session.scalars(select(SpecChunk)).all()
        for result in rag_answer("outlet count north wall", list(chunks), top_k=5)["citations"]:
            print(result)


if __name__ == "__main__":
    main()
