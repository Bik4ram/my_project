module i2c_master (
    input clk, rst, start,
    input [6:0] address,
    input rd_wr,
    input [7:0] din,
    input stop,
    output reg [7:0] dout,
    inout SDA,
    output SCL
);
    parameter READ = 0, WRITE = 1;
    enum bit[3:0] {IDLE, START, ADDR, ADDR_REG, WACK, SACK, WRITE_DATA, WRITE_DATA_REG, WSTOP_1, READ_DATA, READ_DATA_1, READ_DATA_2, READ_DATA_REG, WSTOP_2, STOP} state;
    bit mode;
    logic ack;
    logic [7:0] tx_reg;
    logic [2:0] count;
    logic sda, scl;

    assign SCL = scl;
    assign SDA = (mode) ? 1'bz : sda;

    always @posedge clk or posedge rst begin
        if (rst) begin
            state <= IDLE;
            dout <= 8'b0;
            mode <= 1'b0;
            ack <= 1'b0;
            count <= 3'b0;
            tx_reg <= 8'b0;
            sda <= 1'b1;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= START;
                        mode <= rd_wr;
                    end
                end
                START: begin
                    sda <= 1'b0;
                    #1;
                    sda <= 1'b1;
                    state <= ADDR;
                end
                ADDR: begin
                    // Drive address bits on SDA
                    sda <= address[6];
                    #1;
                    sda <= address[5];
                    #1;
                    sda <= address[4];
                    #1;
                    sda <= address[3];
                    #1;
                    sda <= address[2];
                    #1;
                    sda <= address[1];
                    #1;
                    sda <= address[0];
                    #1;
                    sda <= rd_wr;
                    #1;
                    state <= ADDR_REG;
                end
                ADDR_REG: begin
                    ack <= (mode == WRITE) ? 1'b0 : 1'b1; // ACK for write, NACK for read setup
                    state <= (mode == WRITE) ? WACK : SACK;
                end
                WACK: begin
                    if (ack) state <= WRITE_DATA;
                    else state <= WSTOP_1;
                end
                SACK: begin
                    // For read, sample first bit
                    state <= WRITE_DATA;
                end
                WRITE_DATA: begin
                    sda <= tx_reg[7];
                    #1;
                    sda <= tx_reg[6];
                    #1;
                    sda <= tx_reg[5];
                    #1;
                    sda <= tx_reg[4];
                    #1;
                    sda <= tx_reg[3];
                    #1;
                    sda <= tx_reg[2];
                    #1;
                    sda <= tx_reg[1];
                    #1;
                    sda <= tx_reg[0];
                    #1;
                    tx_reg <= tx_reg << 1;
                    count <= count + 1;
                    if (count == 3'd7) state <= WRITE_DATA_REG;
                end
                WRITE_DATA_REG: begin
                    state <= WSTOP_1;
                end
                WSTOP_1: begin
                    sda <= 1'b1;
                    #1;
                    sda <= 1'b0;
                    state <= STOP;
                end
                STOP: begin
                    sda <= 1'b1;
                    #1;
                    state <= IDLE;
                    dout <= tx_reg;
                    if (mode == READ) begin
                        // Data already captured in READ_DATA flow
                    end
                end
                READ_DATA: begin
                    // Sample SDA into tx_reg
                    tx_reg <= {tx_reg[6:0], sda};
                    count <= count + 1;
                    if (count == 3'd7) state <= READ_DATA_1;
                end
                READ_DATA_1: begin
                    state <= READ_DATA_2;
                end
                READ_DATA_2: begin
                    state <= READ_DATA_REG;
                end
                READ_DATA_REG: begin
                    dout <= tx_reg;
                    state <= WSTOP_2;
                end
                WSTOP_2: begin
                    sda <= 1'b1;
                    #1;
                    sda <= 1'b0;
                    state <= STOP;
                end
            endcase
        end
    end
endmodule : i2c_master