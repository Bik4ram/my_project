`ifndef SV_GENERATED_LINT_STUB_SVH
`define SV_GENERATED_LINT_STUB_SVH

`define uvm_object_utils(T) \
    typedef T type_id; \
    static function T create(string name = ""); \
        T obj; obj = new(name); return obj; \
    endfunction

`define uvm_component_utils(T) \
    typedef T type_id; \
    static function T create(string name = ""); \
        T obj; obj = new(name, null); return obj; \
    endfunction

class uvm_object;
    function new(string name = "");
    endfunction
endclass

class uvm_phase extends uvm_object;
    function new(string name = "uvm_phase");
        super.new(name);
    endfunction
    task raise_objection(uvm_object obj); endtask
    task drop_objection(uvm_object obj); endtask
endclass

class uvm_sequence_item extends uvm_object;
    function new(string name = "uvm_sequence_item");
        super.new(name);
    endfunction
endclass

class uvm_component extends uvm_object;
    function new(string name = "uvm_component", uvm_component parent = null);
        super.new(name);
    endfunction
    virtual function void build_phase(uvm_phase phase); endfunction
    virtual task run_phase(uvm_phase phase); endtask
endclass

class uvm_test extends uvm_component;
    function new(string name = "uvm_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction
endclass

class uvm_sequencer_base extends uvm_component;
    function new(string name = "uvm_sequencer_base", uvm_component parent = null);
        super.new(name, parent);
    endfunction
endclass

class uvm_sequence #(type REQ = uvm_sequence_item) extends uvm_object;
    function new(string name = "uvm_sequence");
        super.new(name);
    endfunction
    task start_item(REQ item); endtask
    task finish_item(REQ item); endtask
    task start(uvm_sequencer_base seqr); endtask
endclass

class i2c_master_sequence_item extends uvm_sequence_item;
    `uvm_object_utils(i2c_master_sequence_item)
    rand bit rd_wr;
    rand bit [6:0] address;
    function new(string name = "i2c_master_sequence_item");
        super.new(name);
    endfunction
endclass

class i2c_master_base_seq extends uvm_sequence #(i2c_master_sequence_item);
    `uvm_object_utils(i2c_master_base_seq)
    i2c_master_sequence_item item;
    function new(string name = "i2c_master_base_seq");
        super.new(name);
    endfunction
endclass

class i2c_master_sequencer extends uvm_sequencer_base;
    `uvm_component_utils(i2c_master_sequencer)
    function new(string name = "i2c_master_sequencer", uvm_component parent = null);
        super.new(name, parent);
    endfunction
endclass

class i2c_master_agent;
    i2c_master_sequencer sqr;
endclass

class i2c_master_environment;
    i2c_master_agent mstr_agnt;
endclass

class i2c_master_base_test extends uvm_test;
    `uvm_component_utils(i2c_master_base_test)
    i2c_master_environment mstr_env;
    function new(string name = "i2c_master_base_test", uvm_component parent = null);
        super.new(name, parent);
    endfunction
endclass

`endif