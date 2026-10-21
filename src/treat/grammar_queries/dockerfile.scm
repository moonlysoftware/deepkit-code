; Minimal definition/reference query for v1.md §6.1; authored for T2.10.
(from_instruction as: (image_alias) @name) @definition.module
(arg_instruction name: (unquoted_string) @name) @definition.constant
(env_pair name: (unquoted_string) @name) @definition.constant
(expansion (variable) @name) @reference.constant
