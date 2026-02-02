from fastapi import FastAPI

from acc.api.routes.compliance import router as compliance_router

app = FastAPI(title="Automated Compliance Check")

app.include_router(compliance_router)
