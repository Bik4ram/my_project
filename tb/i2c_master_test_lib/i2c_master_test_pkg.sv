package i2c_master_test_lib;

    import uvm_pkg::*;
    import i2c_master_pkg::*;
    import i2c_master_seq_lib::*;

    // READ testcase
    class i2c_master_read_test extends uvm_test;
        `uvm_component_utils(i2c_master_read_test)

        function new(string name = "i2c_master_read_test", uvm_component parent);
            super.new(name, parent);
        endfunction : new

        virtual task run_phase(uvm_phase phase);
            i2c_master_read_seq read_seq;
            phase.raise_objection(this);
            read_seq = i2c_master_read_seq::type_id::create("read_seq");
            read_seq.start(null);
            #100;
            phase.drop_objection(this);
        endtask : run_phase
    endclass : i2c_master_read_test

endpackage : i2c_master_test_lib