package i2c_master_pkg;
    import uvm_pkg::*;
    import i2c_master_param_pkg::*;

    typedef struct packed {
        logic rst;
        logic start;
        logic [ADDR_WIDTH-1:0] address;
        logic rd_wr;
        logic [DATA_WIDTH-1:0] din;
        logic [DATA_WIDTH-1:0] dout;
        logic stop;
        bit [3:0] state;
    } i2c_master_sequence_item;

    // Sequence item macros
    `define I2C_MASTER_SEQ_ITEM_DEFAULT \
        {1'b0, 1'b0, {ADDR_WIDTH{1'b0}}, 1'b0, {DATA_WIDTH{1'b0}}, {DATA_WIDTH{1'b0}}, 1'b0, {4{1'b0}}}

    `uvm_utils_begin(i2c_master_sequence_item)
        `uvm_field_object(rst, UVM_ALL_ON)
        `uvm_field_start
        `uvm_field_bit(rst, UVM_ALL_ON)
        `uvm_field_bit(start, UVM_ALL_ON)
        `uvm_field_enum(logic [ADDR_WIDTH-1:0], address, UVM_ALL_ON)
        `uvm_field_bit(rd_wr, UVM_ALL_ON)
        `uvm_field_data(din, UVM_ALL_ON)
        `uvm_field_data(dout, UVM_ALL_ON)
        `uvm_field_bit(stop, UVM_ALL_ON)
        `uvm_field_enum_bit(state, UVM_ALL_ON)
    `uvm_utils_end

    // Convenience function
    function new(string name = "i2c_master_sequence_item");
        super.new(name);
        rst = 1'b0;
        start = 1'b0;
        address = {ADDR_WIDTH{1'b0}};
        rd_wr = 1'b0;
        din = {DATA_WIDTH{1'b0}};
        dout = {DATA_WIDTH{1'b0}};
        stop = 1'b0;
        state = IDLE;
    endfunction : new
endpackage : i2c_master_pkg