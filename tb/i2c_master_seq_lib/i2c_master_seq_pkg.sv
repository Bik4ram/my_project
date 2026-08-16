package i2c_master_seq_lib;

    import uvm_pkg::*;
    import i2c_master_pkg::*;
    import i2c_master_param_pkg::*;

    // Write sequence
    class i2c_master_write_seq extends uvm_sequence #(i2c_master_sequence_item);
        `uvm_object_utils(i2c_master_write_seq)

        function new(string name = "i2c_master_write_seq");
            super.new(name);
        endfunction : new

        task body();
            i2c_master_sequence_item item;
            item = i2c_master_sequence_item::type_id::create("item");
            start_item(item);

            // Write transaction: address with rd_wr=WRITE, then data
            item.rst = 1'b0;
            item.start = 1'b1;
            item.address = 7'h50;     // Example slave address
            item.rd_wr = WRITE;
            item.din = 8'hA5;         // Example write data
            item.stop = 1'b0;

            finish_item(item);
            `uvm_info("I2C_WRITE", $sformatf("Wrote address=%0h data=%0h", item.address, item.din), UVM_MEDIUM)
        endtask : body
    endclass : i2c_master_write_seq

    // READ sequence - the testcase to create
    class i2c_master_read_seq extends uvm_sequence #(i2c_master_sequence_item);
        `uvm_object_utils(i2c_master_read_seq)

        function new(string name = "i2c_master_read_seq");
            super.new(name);
        endfunction : new

        task body();
            i2c_master_sequence_item item;
            item = i2c_master_sequence_item::type_id::create("item");
            start_item(item);

            // Read transaction: address with rd_wr=READ, then capture data
            item.rst = 1'b0;
            item.start = 1'b1;
            item.address = 7'h50;     // Example slave address
            item.rd_wr = READ;
            item.din = 8'h00;
            item.stop = 1'b1;

            finish_item(item);
            `uvm_info("I2C_READ", $sformatf("Read address=%0h, expecting data on dout", item.address), UVM_MEDIUM)
        endtask : body
    endclass : i2c_master_read_seq

endpackage : i2c_master_seq_lib