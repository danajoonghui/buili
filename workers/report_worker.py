from services.api.buili.database import SessionLocal, init_db
from services.api.buili.reports import build_report


def main() -> None:
    init_db()
    with SessionLocal() as session:
        for project_id, in session.execute("select project_id from projects").all():
            report_id, path = build_report(session, project_id, "punch", "pdf")
            print(report_id, path)


if __name__ == "__main__":
    main()

