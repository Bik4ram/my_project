package i2c_master_agent_pkg;
    import uvm_pkg::*;
    import i2c_master_pkg::*;
    import i2c_master_param_pkg::*;

    class i2c_master_agent_config extends uvm_object;
        `uvm_object_utils(i2c_master_agent_config)

        virtual i2c_master_if      vif_confg;
        is_active_enum               is_active = UVM_ACTIVE;

        function new(string name = "i2c_master_agent_config");
            super.new(name);
        endfunction : new
    endclass : i2c_master_agent_config

endpackage : i2c_master_agent_pkg