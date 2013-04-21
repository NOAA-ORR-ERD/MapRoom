from Mock_file import Mock_file


class Mock_imp:
    def load_module( self, name, file, path, suffixes ):
        globals = {}
        locals = {}

        eval( compile(
            Mock_file( path ).read(),
            path,
            "exec",
        ), globals, locals )

        return locals.get( name )

    def get_suffixes( self ):
        return [
            ( ".so", "rb", 3 ),
            ( "module.so", "rb", 3 ),
            ( ".py", "U", 1 ),
            ( ".pyc", "rb", 2 ),
        ]
