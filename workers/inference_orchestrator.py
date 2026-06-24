import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from services.api.buili.database import SessionLocal, init_db
from services.api.buili.models import Job
from services.api.buili.pipeline import run_analysis_job


def main() -> None:
    init_db()
    with SessionLocal() as session:
        jobs = session.query(Job).filter(Job.state == "queued").all()
        for job in jobs:
            run_analysis_job(job.job_id, session)
            print(f"processed {job.job_id}")


if __name__ == "__main__":
    main()

