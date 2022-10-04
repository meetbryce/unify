
from .db_wrapper import (
    ClickhouseWrapper, 
    DBManager, 
    DuckDBWrapper
)

DBMGR_CLASS: DBManager = ClickhouseWrapper if os.environ['DATABASE_BACKEND'] == 'clickhouse' else DuckDBWrapper
UNIFY_META_SCHEMA = 'unify_schema'

Base = declarative_base()

def uniq_id():
    return str(uuid.uuid4())

class SchemaType(enum.Enum):
    adapter = 1
    connection = 2

class Schemata(Base):  # type: ignore
    __tablename__ = "schemata"

    id = Column(String, default=uniq_id, primary_key=True)
    name = Column(String)
    type = Column(String)
    type_or_spec = Column(Enum(SchemaType))
    created = Column(DateTime, default=datetime.utcnow())
    comment = Column(String)

    __table_args__ = DBMGR_CLASS.get_sqlalchemy_table_args(primary_key='id', schema=UNIFY_META_SCHEMA)

def create_tables():
    uri = 'clickhouse://' + \
        os.environ['DATABASE_USER'] + ':' + '@' +\
        os.environ['DATABASE_HOST'] + '/default'
    print(uri)
    engine = create_engine(uri)
    Base.metadata.create_all(engine)
