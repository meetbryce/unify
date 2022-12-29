
import sqlalchemy.engine.cursor

def unify_patched_inserted_primary_key(self):
    if self.context.executemany:
        return self.inserted_primary_key_rows[0]

    ikp = self.inserted_primary_key_rows
    if ikp:
        return ikp[0]
    else:
        return None

def patch_sqlalchemy_cursor():
    setattr(sqlalchemy.engine.cursor.BaseCursorResult, '__orig_ipk_m',
        sqlalchemy.engine.cursor.BaseCursorResult.inserted_primary_key)
    setattr(sqlalchemy.engine.cursor.BaseCursorResult, 'inserted_primary_key',
        property(unify_patched_inserted_primary_key))
