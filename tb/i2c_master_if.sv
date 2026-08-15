interface i2c_master_if (
    input logic clk,
    input logic rst,
    output logic start,
    output logic [6:0] address,
    output logic rd_wr,
    output logic [7:0] din,
    output logic stop,
    input logic [7:0] dout,
    inout SDA,
    output SCL
);
    modport master (
        input clk,
        input rst,
        output start,
        output address,
        output rd_wr,
        output din,
        output stop,
        input dout,
        inout SDA,
        output SCL
    );

    modport slave (
        input clk,
        input rst,
        input start,
        input address,
        input rd_wr,
        input din,
        input stop,
        output dout,
        inout SDA,
        input SCL
    );
endinterface : i2c_master_if