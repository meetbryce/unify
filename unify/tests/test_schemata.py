from sqlalchemy.orm.session import Session

from unify.db_wrapper import Schemata

from unify.db_wrapper import dbmgr

def test_schemata():
    with dbmgr() as db:
        session = Session(bind=db.engine)

        github = Schemata(name="github", type="connection", description="Access to Github")
        jiraa = Schemata(name="jira_adapter", type="adapter", description="Access to JIRA")

        session.add(github)
        session.add(jiraa)
        session.commit()
            