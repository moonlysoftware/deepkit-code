; Minimal definition/reference query for v1.md §6.1; authored for T2.10.
(function_declaration name: (identifier) @name) @definition.function
(class_declaration name: (identifier) @name) @definition.class
(call_expression . (identifier) @name) @reference.call
