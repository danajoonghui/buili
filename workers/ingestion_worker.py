from services.api.buili.database import SessionLocal, init_db
from services.api.buili.pipeline import parse_pdf_document
from services.api.buili.models import Document


def main() -> None:
    init_db()
    with SessionLocal() as session:
        for doc in session.query(Document).all():
            pages = parse_pdf_document(doc)
            print(f"{doc.doc_id}: parsed {len(pages)} pages")


if __name__ == "__main__":
    main()

