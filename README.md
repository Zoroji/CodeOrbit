"# CodeOrbit" 





########### CHUNKING STRATEGY #######################

We use **Tree-sitter** for polyglot semantic code chunking (Python, JavaScript, TypeScript, TSX). 
Tree-sitter builds a concrete syntax tree (CST) that pinpoints exact byte ranges and line boundaries for every function, class, import, and statement across multiple languages.

### PSEUDO CODE FOR CHUNKING (TREE-SITTER VERSION)

```python
START

# STEP 1: INITIALIZE & PARSER SELECTION
READ source file as bytes
SELECT grammar dynamically from file extension (.py, .js, .jsx, .ts, .tsx)
INITIALIZE Tree-sitter Parser with selected Language
PARSE source_bytes into Tree-sitter Syntax Tree (root_node)

CREATE empty collections:
    imports_nodes
    globals_nodes
    output_chunks

SET TARGET_CHUNK_SIZE = 500 tokens

# STEP 2: PROCESS ROOT SYNTAX NODES
FOR EACH child node in root_node.children (where child is named):

    # A. IMPORTS (Python & JS/TS)
    IF child.type in ["import_statement", "import_from_statement"]:
        ADD child to imports_nodes

    # B. CLASSES (class_definition / class_declaration)
    ELSE IF child.type in ["class_definition", "class_declaration"]:

        EXTRACT class name from child.child_by_field_name("name")
        EXTRACT docstring / leading comments
        EXTRACT class-level data members (attributes / field_definitions)
        FIND method nodes inside class body (function_definition / method_definition)

        # Build interface summary without method implementations
        FOR EACH method node:
            EXTRACT method signature line only (e.g. "def foo(...): ...")
            DO NOT include method body

        CREATE CLASS OVERVIEW chunk containing:
            class name
            docstring / comments
            class-level attributes
            method signature stubs

        ADD CLASS OVERVIEW to output_chunks

        # Store complete method implementations separately
        FOR EACH method node inside class body:
            EXTRACT complete method source code
            ATTACH parent = class name

            IF method tokens <= TARGET_CHUNK_SIZE:
                CREATE METHOD chunk (index 1 of 1)
                ADD to output_chunks
            ELSE:
                SPLIT method statements recursively at block/statement boundaries
                FOR EACH split statement group:
                    CREATE METHOD chunk (index i of total)
                    ADD to output_chunks

    # C. TOP-LEVEL FUNCTIONS (function_definition / function_declaration / arrow_function)
    ELSE IF child.type in ["function_definition", "function_declaration", "arrow_function"]:

        EXTRACT complete function source code

        IF function tokens <= TARGET_CHUNK_SIZE:
            CREATE FUNCTION chunk (index 1 of 1)
            ADD to output_chunks
        ELSE:
            SPLIT function statements recursively at block/statement boundaries
            FOR EACH split statement group:
                CREATE FUNCTION chunk (index i of total)
                ADD to output_chunks

    # D. GLOBAL VARIABLES & CONSTANTS
    ELSE IF child.type in ["expression_statement", "lexical_declaration", "variable_declaration"]:
        IF child represents assignment / variable declaration:
            ADD child to globals_nodes

# STEP 3: CREATE AGGREGATED FILE-LEVEL CHUNKS
IF imports_nodes is not empty:
    COMBINE all import statements into one single IMPORTS chunk
    ADD to output_chunks

IF globals_nodes is not empty:
    COMBINE all global variable assignments into one single GLOBALS chunk
    ADD to output_chunks

# STEP 4: RETURN RESULTS
RETURN output_chunks

END
```
