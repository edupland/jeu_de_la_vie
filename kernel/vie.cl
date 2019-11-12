#include "kernel/common.cl"

#define PIX_BLOC 32

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// vie
////////////////////////////////////////////////////////////////////////////////

__kernel void vie (__global unsigned *in, __global unsigned *out) {

    __local unsigned tile[TILEY+2][TILEX+2];

    int x = get_global_id (0);
    int y = get_global_id (1);
    int xloc = get_local_id (0);
    int yloc = get_local_id (1);
    int haut, bas, gauche, droite;

    haut = (xloc == 0 && x > 0);
    bas = (xloc == TILEX - 1 && x < DIM - 1);
    gauche = (yloc == 0 && y > 0);
    droite = (yloc == TILEY - 1 && y < DIM - 1);

    tile [yloc + 1][xloc+1] = in [y * DIM + x];

    unsigned result = tile [yloc + 1][xloc+1];

    if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1){

        if (haut || bas)
            tile [ yloc + 1][ xloc + 1 - haut + bas ] = in [ y * DIM + x - haut + bas ];

        if (gauche || droite)
            tile [ yloc + 1 - gauche + droite ][ xloc + 1] = in [( y - gauche + droite ) * DIM + x ];

        if ((haut || bas) && (gauche || droite))
            tile [ yloc + 1 - gauche + droite ][ xloc + 1 - haut + bas ] = in [( y - gauche + droite ) * DIM + x - haut + bas ];

        barrier (CLK_LOCAL_MEM_FENCE);

        unsigned n = 0;
        n += (tile[yloc][xloc] != 0);
        n += (tile[yloc][xloc+1] != 0);
        n += (tile[yloc][xloc+2] != 0);
        n += (tile[yloc+1][xloc] != 0);
        n += (tile[yloc+1][xloc+2] != 0);
        n += (tile[yloc+2][xloc] != 0);
        n += (tile[yloc+2][xloc+1] != 0);
        n += (tile[yloc+2][xloc+2] != 0);
            
        if (tile[yloc+1][xloc+1] != 0)
            result = (n == 2 || n == 3) * 0xFFFF00FF;
        else
            result = (n == 3) * 0xFFFF00FF;
        
    }
        
    out [y * DIM + x] = result;
}
 