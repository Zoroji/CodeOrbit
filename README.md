"# CodeOrbit" 





###########CHUNKING STRATEGY#######################

we are using AST since for code we dont want to have basic Text base chunking 

AST provides use the line of code where a function start and where it ends 

HERE IS THE PSEUDO CODE FOR CHUNKING VERSION 1 

```python
START

# STEP 1: INITIALIZE
READ Python source file
PARSE source into AST

CREATE empty lists:
    imports
    global_variables
    output_chunks

# STEP 2: PROCESS TOP-LEVEL NODES
FOR EACH node in AST top-level nodes:

    IF node is an IMPORT:
        ADD import statement to imports

    ELSE IF node is a GLOBAL VARIABLE assignment:
        ADD variable assignment to global_variables

    ELSE IF node is a CLASS:

        EXTRACT class name
        EXTRACT class docstring, if present
        EXTRACT class-level data members
        FIND methods directly inside the class

        FOR EACH method:
            EXTRACT method signature only
            DO NOT include method implementation

        CREATE CLASS OVERVIEW chunk containing:
            class name
            class docstring
            class-level data members
            method signatures

        ADD CLASS OVERVIEW to output_chunks

        # Store method implementations separately
        FOR EACH method inside the class:

            EXTRACT complete method source
            INCLUDE decorators, if any

            CREATE METHOD chunk containing:
                method name
                parent class name
                method implementation
                source location

            ADD METHOD chunk to output_chunks

    ELSE IF node is a TOP-LEVEL FUNCTION:

        EXTRACT complete function source
        INCLUDE decorators, if any

        CREATE FUNCTION chunk containing:
            function name
            function implementation
            source location

        ADD FUNCTION chunk to output_chunks

# STEP 3: CREATE FILE-LEVEL CHUNKS
IF imports is not empty:
    CREATE one IMPORTS chunk
    ADD it to output_chunks

IF global_variables is not empty:
    CREATE one GLOBAL VARIABLES chunk
    ADD it to output_chunks

# STEP 4: RETURN RESULTS
RETURN output_chunks

END
```
