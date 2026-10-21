(struct_specifier name: (type_identifier) @name body:(_)) @definition.class

(declaration type: (union_specifier name: (type_identifier) @name)) @definition.class

(function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function

(function_definition declarator: (function_declarator declarator: (field_identifier) @name)) @definition.function

(function_definition declarator: (function_declarator declarator: (qualified_identifier scope: (namespace_identifier) @local.scope name: (identifier) @name))) @definition.method

(type_definition declarator: (type_identifier) @name) @definition.type

(enum_specifier name: (type_identifier) @name) @definition.type

(class_specifier name: (type_identifier) @name) @definition.class

(call_expression function: (identifier) @name) @reference.call

(function_definition declarator: (pointer_declarator declarator: (function_declarator declarator: (identifier) @name))) @definition.function

(function_definition declarator: (pointer_declarator declarator: (pointer_declarator declarator: (function_declarator declarator: (identifier) @name)))) @definition.function

(function_definition declarator: (reference_declarator (function_declarator declarator: (identifier) @name))) @definition.function
