from sqlalchemy.orm.session import Session

from unify.db_wrapper import Schemata, SchemataType

from unify import dbmgr

def test_schemata():
    with dbmgr() as db:
        session = Session(bind=db.engine)

        github = Schemata(name="github", type=SchemataType.connection, comment="Access to Github")
        jiraa = Schemata(name="jira_adapter", type=SchemataType.adapter, comment="Access to JIRA")

        session.add(github)
        session.add(jiraa)
        session.commit()
            