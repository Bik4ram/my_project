package i2c_master_param_pkg;
    parameter READ = 0, WRITE = 1;
    parameter ADDR_WIDTH = 7;
    parameter DATA_WIDTH = 8;
    parameter PERIOD = 100; // ns

    typedef struct packed {
        logic [ADDR_WIDTH-1:0] address;
        logic rd_wr;
        logic [DATA_WIDTH-1:0] din;
        logic stop;
    } i2c_master_transaction_s;
endpackage : i2c_master_param_pkg