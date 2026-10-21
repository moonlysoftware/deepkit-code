; Minimal definition/reference query for v1.md §6.1; authored for T2.10.
(function_declaration name: (identifier) @name) @definition.function
(variable_declaration . (identifier) @name) @definition.constant
(call_expression function: (identifier) @name) @reference.call
