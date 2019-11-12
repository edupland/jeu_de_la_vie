
#include "compute.h"
#include "debug.h"
#include "global.h"
#include "graphics.h"
#include "ocl.h"
#include "scheduler.h"

#ifdef ENABLE_MPI
	#include "mpi.h"
#endif

#include <stdbool.h>

static int compute_new_state_old(int y, int x)
{
	unsigned n = 0;
	unsigned change = 0;

	if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1)
	{
		for (int i = y - 1; i <= y + 1; i++)
			for (int j = x - 1; j <= x + 1; j++)
				if (i != y || j != x)
					n += (cur_img(i, j) != 0);

		if (cur_img(y, x) != 0)
		{
			if (n == 2 || n == 3)
				n = 0xFFFF00FF;
			else
			{
				n = 0;
				change = 1;
			}
		}
		else
		{
			if (n == 3)
			{
				n = 0xFFFF00FF;
				change = 1;
			}
			else
				n = 0;
		}

		next_img(y, x) = n;
	}

	return change;
}

// Compute new_state avec moins de sauts conditionnels
static int compute_new_state(int y, int x)
{
	unsigned n = 0;
	unsigned change = 0;

	n += (cur_img(y - 1, x - 1) != 0);
	n += (cur_img(y - 1, x) != 0);
	n += (cur_img(y - 1, x + 1) != 0);
	n += (cur_img(y, x - 1) != 0);
	n += (cur_img(y, x + 1) != 0);
	n += (cur_img(y + 1, x - 1) != 0);
	n += (cur_img(y + 1, x) != 0);
	n += (cur_img(y + 1, x + 1) != 0);

	// On laisse un 'if' pour éviter 2 appels à cur_img
	if (cur_img(y, x) != 0){
		n = (n == 2 || n == 3) * 0xFFFF00FF;
	}else{
		n = (n == 3) * 0xFFFF00FF;
	}

	change = (cur_img(y, x) != n);

	next_img(y, x) = n;

	return change;
}


// ============================== Version séquentielle d'origine ==============================

static int traiter_tuile(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state_old(i, j);

	return change;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned vie_compute_seq(unsigned nb_iter)
{
	for (unsigned it = 1; it <= nb_iter; it++)
	{

		// On traite toute l'image en un coup (oui, c'est une grosse tuile)
		unsigned change = traiter_tuile(0, 0, DIM - 1, DIM - 1);

		swap_images();

		if (!change)
			return it;
	}

	return 0;
}


// ============================== Version séquentielle de base ==============================

unsigned vie_compute_seq_base(unsigned nb_iter)
{

	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		for (int i = 1; i < DIM-1; i++){
			for (int j = 1; j < DIM-1; j++){
				change |= compute_new_state(i, j);
			}
		}

		swap_images();

		if (!change)
			return it;

	}

	return 0;
}



// ============================== Version séquentielle tuilée ==============================

static int traiter_tuile_seq_tiled(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}

static unsigned tranche = 0;

unsigned vie_compute_seq_tiled(unsigned nb_iter)
{

	tranche = DIM / GRAIN;
	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		for (int i = 0; i < GRAIN; i++){
			for (int j = 0; j < GRAIN; j++){
				change |= traiter_tuile_seq_tiled(
					(i == 0) + i * tranche,
					(j == 0) + j * tranche,
					(i + 1) * tranche - 1 - (i == GRAIN-1),
					(j + 1) * tranche - 1 - (j == GRAIN-1));
			}
		}

		swap_images();

		if (!change)
			return it;
	}

	return 0;
}


// ============================== Version séquentielle tuilée optimisée ==============================

static int traiter_tuile_seq_tiled_opt(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}

unsigned vie_compute_seq_tiled_opt(unsigned nb_iter)
{

	// true si la tuile a été modifiée
	bool tab[GRAIN][GRAIN];
	for (int i = 0; i < GRAIN; i++)
		for (int j = 0 ; j < GRAIN ; j++)
			tab[i][j] = false;

	tranche = DIM / GRAIN;
	unsigned change = 0;


	// 1er tour pour initialiser les tuiles modifiées dans le tableau

	change |= traiter_tuile_seq_tiled_opt(1, 1, tranche, DIM-2);
	change |= traiter_tuile_seq_tiled_opt(DIM-tranche, 1, DIM-2, DIM -2);
	change |= traiter_tuile_seq_tiled_opt(tranche+1, 1, DIM-tranche-1, tranche);
	change |= traiter_tuile_seq_tiled_opt(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

	for (int i = 1 ; i < GRAIN-1; i++){
		for (int j = 1 ; j < GRAIN-1; j++){
			change |= traiter_tuile_seq_tiled_opt(
				i * tranche,
				j * tranche,
				(i + 1) * tranche - 1,
				(j + 1) * tranche - 1
			);

			tab[i][j] = change;
		}
	}
	
	for (unsigned it = 1; it <= nb_iter; it++){

		// On traite les bords de l'image
		change |= traiter_tuile_seq_tiled_opt(1, 1, tranche, DIM-2);
		change |= traiter_tuile_seq_tiled_opt(DIM-tranche, 1, DIM-2, DIM -2);
		change |= traiter_tuile_seq_tiled_opt(tranche+1, 1, DIM-tranche-1, tranche);
		change |= traiter_tuile_seq_tiled_opt(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

		for (int i = 1 ; i < GRAIN-1; i++){
			for (int j = 1 ; j < GRAIN-1; j++){
				
				// Si une des 8 tuiles alentours ont été modifiées
				if (!tab[i][j] && !tab[i-1][j] && !tab[i+1][j] && !tab[i][j-1] && !tab[i][j+1] && !tab[i+1][j+1] && !tab[i-1][j-1] && !tab[i+1][j-1] && !tab[i-1][j+1]){
					change |= traiter_tuile_seq_tiled_opt(
						i * tranche,
						j * tranche,
						(i + 1) * tranche - 1,
						(j + 1) * tranche - 1
					);

					tab[i][j] = change;
				}
			}
		}

		swap_images();

		if (!change)
		{
			return it;
		}
	}

	return 0;
}


// ============================== Version parallèle de base statique ==============================

unsigned vie_compute_omp_base_static(unsigned nb_iter)
{

	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		#pragma omp parallel for schedule (static) reduction(|:change)
		for (int i = 1; i < DIM-1; i++){
			for (int j = 1; j < DIM-1; j++){
				change |= compute_new_state(i, j);
			}
		}

		swap_images();

		if (!change)
			return it;
			
	}

	return 0;
}


// ============================== Version parallèle de base cyclique ==============================

unsigned vie_compute_omp_base_cyclic(unsigned nb_iter)
{

	unsigned change = 0;
	
	for (unsigned it = 1; it <= nb_iter; it++){

		#pragma omp parallel for schedule (static, 2) reduction(|:change)
		for (int i = 1; i < DIM-1; i++){
			for (int j = 1; j < DIM-1; j++){
				change |= compute_new_state(i, j);
			}
		}

		swap_images();

		if (!change)
			return it;
			
	}

	return 0;
}


// ============================== Version parallèle de base dynamique ==============================

unsigned vie_compute_omp_base_dynamic(unsigned nb_iter)
{

	unsigned change = 0 ; 
	
	for (unsigned it = 1; it <= nb_iter; it++){

		#pragma omp parallel for schedule (dynamic, 1) reduction(|:change)
		for (int i = 1; i < DIM-1; i++){
			for (int j = 1; j < DIM-1; j++){
				change |= compute_new_state(i, j);
			}
		}

		swap_images();

		if (!change)
			return it;
			
	}

	return 0;
}


// ============================== Version parallèle de base avec collapse ==============================

unsigned vie_compute_omp_base_collapse(unsigned nb_iter)
{

	unsigned change = 0 ; 
	
	for (unsigned it = 1; it <= nb_iter; it++){

		#pragma omp parallel for collapse(2) schedule(static) reduction(|:change)
		for (int i = 1; i < DIM-1; i++){
			for (int j = 1; j < DIM-1; j++){
				change |= compute_new_state(i, j);
			}
		}

		swap_images();

		if (!change)
			return it;
			
	}

	return 0;
}


// ============================== Version parallèle tuilée statique ==============================

static int traiter_tuile_omp_tiled_static(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for schedule(static) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_static(unsigned nb_iter)
{

	tranche = DIM / GRAIN;
	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		for (int i = 0; i < GRAIN; i++){
			for (int j = 0; j < GRAIN; j++){
				change |= traiter_tuile_omp_tiled_static(
					(i == 0) + i * tranche,
					(j == 0) + j * tranche,
					(i + 1) * tranche - 1 - (i == GRAIN-1),
					(j + 1) * tranche - 1 - (j == GRAIN-1));
			}
		}

		swap_images();

		if (!change)
			return it;
	}

	return 0;
}


// ============================== Version parallèle tuilée cyclique ==============================

static int traiter_tuile_omp_tiled_cyclic(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for schedule(static, 1) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_cyclic(unsigned nb_iter)
{

	tranche = DIM / GRAIN;
	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		for (int i = 0; i < GRAIN; i++){
			for (int j = 0; j < GRAIN; j++){
				change |= traiter_tuile_omp_tiled_cyclic(
					(i == 0) + i * tranche,
					(j == 0) + j * tranche,
					(i + 1) * tranche - 1 - (i == GRAIN-1),
					(j + 1) * tranche - 1 - (j == GRAIN-1));
			}
		}

		swap_images();

		if (!change)
			return it;
	}

	return 0;
}


// ============================== Version parallèle tuilée dynamique ==============================

static int traiter_tuile_omp_tiled_dynamic(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for schedule(dynamic, 1) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_dynamic(unsigned nb_iter)
{

	tranche = DIM / GRAIN;
	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		for (int i = 0; i < GRAIN; i++){
			for (int j = 0; j < GRAIN; j++){
				change |= traiter_tuile_omp_tiled_dynamic(
					(i == 0) + i * tranche,
					(j == 0) + j * tranche,
					(i + 1) * tranche - 1 - (i == GRAIN-1),
					(j + 1) * tranche - 1 - (j == GRAIN-1));
			}
		}

		swap_images();

		if (!change)
			return it;
	}

	return 0;
}


// ============================== Version parallèle tuilée avec collapse ==============================

static int traiter_tuile_omp_tiled_collapse(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for collapse(2) schedule(static) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_collapse(unsigned nb_iter)
{

	tranche = DIM / GRAIN;
	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		for (int i = 0; i < GRAIN; i++){
			for (int j = 0; j < GRAIN; j++){
				change |= traiter_tuile_omp_tiled_collapse(
					(i == 0) + i * tranche,
					(j == 0) + j * tranche,
					(i + 1) * tranche - 1 - (i == GRAIN-1),
					(j + 1) * tranche - 1 - (j == GRAIN-1));
			}
		}

		swap_images();

		if (!change)
			return it;
	}

	return 0;
}


// ============================== Version parallèle optimisée statique ==============================

static int traiter_tuile_omp_tiled_opt_static(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for schedule(static) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_opt_static(unsigned nb_iter)
{

	bool tab[GRAIN][GRAIN];
	for (int i = 0; i < GRAIN; i++)
		for (int j = 0 ; j < GRAIN ; j++)
			tab[i][j] = false;

	tranche = DIM / GRAIN;
	unsigned change = 0;

	change |= traiter_tuile_omp_tiled_opt_static(1, 1, tranche, DIM-2);
	change |= traiter_tuile_omp_tiled_opt_static(DIM-tranche, 1, DIM-2, DIM -2);
	change |= traiter_tuile_omp_tiled_opt_static(tranche+1, 1, DIM-tranche-1, tranche);
	change |= traiter_tuile_omp_tiled_opt_static(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

	for (int i = 1 ; i < GRAIN-1; i++){
		for (int j = 1 ; j < GRAIN-1; j++){
			change |= traiter_tuile_omp_tiled_opt_static(
				i * tranche,
				j * tranche,
				(i + 1) * tranche - 1,
				(j + 1) * tranche - 1
			);

			tab[i][j] = change;
		}
	}
	
	for (unsigned it = 1; it <= nb_iter; it++){

		change |= traiter_tuile_omp_tiled_opt_static(1, 1, tranche, DIM-2);
		change |= traiter_tuile_omp_tiled_opt_static(DIM-tranche, 1, DIM-2, DIM -2);
		change |= traiter_tuile_omp_tiled_opt_static(tranche+1, 1, DIM-tranche-1, tranche);
		change |= traiter_tuile_omp_tiled_opt_static(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

		for (int i = 1 ; i < GRAIN-1; i++){
			for (int j = 1 ; j < GRAIN-1; j++){
				
				if (!tab[i][j] && !tab[i-1][j] && !tab[i+1][j] && !tab[i][j-1] && !tab[i][j+1] && !tab[i+1][j+1] && !tab[i-1][j-1] && !tab[i+1][j-1] && !tab[i-1][j+1]){
					change |= traiter_tuile_omp_tiled_opt_static(
						i * tranche,
						j * tranche,
						(i + 1) * tranche - 1,
						(j + 1) * tranche - 1
					);

					tab[i][j] = change;
				}
			}
		}

		swap_images();

		if (!change)
		{
			return it;
		}
	}

	return 0;
}


// ============================== Version parallèle optimisée cyclique ==============================

static int traiter_tuile_omp_tiled_opt_cyclic(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for schedule(static, 1) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_opt_cyclic(unsigned nb_iter)
{

	bool tab[GRAIN][GRAIN];
	for (int i = 0; i < GRAIN; i++)
		for (int j = 0 ; j < GRAIN ; j++)
			tab[i][j] = false;

	tranche = DIM / GRAIN;
	unsigned change = 0;

	change |= traiter_tuile_omp_tiled_opt_cyclic(1, 1, tranche, DIM-2);
	change |= traiter_tuile_omp_tiled_opt_cyclic(DIM-tranche, 1, DIM-2, DIM -2);
	change |= traiter_tuile_omp_tiled_opt_cyclic(tranche+1, 1, DIM-tranche-1, tranche);
	change |= traiter_tuile_omp_tiled_opt_cyclic(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

	for (int i = 1 ; i < GRAIN-1; i++){
		for (int j = 1 ; j < GRAIN-1; j++){
			change |= traiter_tuile_omp_tiled_opt_cyclic(
				i * tranche,
				j * tranche,
				(i + 1) * tranche - 1,
				(j + 1) * tranche - 1
			);

			tab[i][j] = change;
		}
	}
	
	for (unsigned it = 1; it <= nb_iter; it++){

		change |= traiter_tuile_omp_tiled_opt_cyclic(1, 1, tranche, DIM-2);
		change |= traiter_tuile_omp_tiled_opt_cyclic(DIM-tranche, 1, DIM-2, DIM -2);
		change |= traiter_tuile_omp_tiled_opt_cyclic(tranche+1, 1, DIM-tranche-1, tranche);
		change |= traiter_tuile_omp_tiled_opt_cyclic(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

		for (int i = 1 ; i < GRAIN-1; i++){
			for (int j = 1 ; j < GRAIN-1; j++){
				
				if (!tab[i][j] && !tab[i-1][j] && !tab[i+1][j] && !tab[i][j-1] && !tab[i][j+1] && !tab[i+1][j+1] && !tab[i-1][j-1] && !tab[i+1][j-1] && !tab[i-1][j+1]){
					change |= traiter_tuile_omp_tiled_opt_cyclic(
						i * tranche,
						j * tranche,
						(i + 1) * tranche - 1,
						(j + 1) * tranche - 1
					);

					tab[i][j] = change;
				}
			}
		}

		swap_images();

		if (!change)
		{
			return it;
		}
	}

	return 0;
}


// ============================== Version parallèle optimisée dynamique ==============================

static int traiter_tuile_omp_tiled_opt_dynamic(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for schedule(dynamic, 1) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_opt_dynamic(unsigned nb_iter)
{

	bool tab[GRAIN][GRAIN];
	for (int i = 0; i < GRAIN; i++)
		for (int j = 0 ; j < GRAIN ; j++)
			tab[i][j] = false;

	tranche = DIM / GRAIN;
	unsigned change = 0;

	change |= traiter_tuile_omp_tiled_opt_dynamic(1, 1, tranche, DIM-2);
	change |= traiter_tuile_omp_tiled_opt_dynamic(DIM-tranche, 1, DIM-2, DIM -2);
	change |= traiter_tuile_omp_tiled_opt_dynamic(tranche+1, 1, DIM-tranche-1, tranche);
	change |= traiter_tuile_omp_tiled_opt_dynamic(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

	for (int i = 1 ; i < GRAIN-1; i++){
		for (int j = 1 ; j < GRAIN-1; j++){
			change |= traiter_tuile_omp_tiled_opt_dynamic(
				i * tranche,
				j * tranche,
				(i + 1) * tranche - 1,
				(j + 1) * tranche - 1
			);

			tab[i][j] = change;
		}
	}
	
	for (unsigned it = 1; it <= nb_iter; it++){

		change |= traiter_tuile_omp_tiled_opt_dynamic(1, 1, tranche, DIM-2);
		change |= traiter_tuile_omp_tiled_opt_dynamic(DIM-tranche, 1, DIM-2, DIM -2);
		change |= traiter_tuile_omp_tiled_opt_dynamic(tranche+1, 1, DIM-tranche-1, tranche);
		change |= traiter_tuile_omp_tiled_opt_dynamic(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

		for (int i = 1 ; i < GRAIN-1; i++){
			for (int j = 1 ; j < GRAIN-1; j++){
				
				if (!tab[i][j] && !tab[i-1][j] && !tab[i+1][j] && !tab[i][j-1] && !tab[i][j+1] && !tab[i+1][j+1] && !tab[i-1][j-1] && !tab[i+1][j-1] && !tab[i-1][j+1]){
					change |= traiter_tuile_omp_tiled_opt_dynamic(
						i * tranche,
						j * tranche,
						(i + 1) * tranche - 1,
						(j + 1) * tranche - 1
					);

					tab[i][j] = change;
				}
			}
		}

		swap_images();

		if (!change)
		{
			return it;
		}
	}

	return 0;
}


// ============================== Version parallèle optimisée avec collapse ==============================

static int traiter_tuile_omp_tiled_opt_collapse(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	#pragma omp parallel for collapse(2) schedule(static) reduction(|:change)
	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}


unsigned vie_compute_omp_tiled_opt_collapse(unsigned nb_iter)
{

	bool tab[GRAIN][GRAIN];
	for (int i = 0; i < GRAIN; i++)
		for (int j = 0 ; j < GRAIN ; j++)
			tab[i][j] = false;

	tranche = DIM / GRAIN;
	unsigned change = 0;

	change |= traiter_tuile_omp_tiled_opt_collapse(1, 1, tranche, DIM-2);
	change |= traiter_tuile_omp_tiled_opt_collapse(DIM-tranche, 1, DIM-2, DIM -2);
	change |= traiter_tuile_omp_tiled_opt_collapse(tranche+1, 1, DIM-tranche-1, tranche);
	change |= traiter_tuile_omp_tiled_opt_collapse(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

	for (int i = 1 ; i < GRAIN-1; i++){
		for (int j = 1 ; j < GRAIN-1; j++){
			change |= traiter_tuile_omp_tiled_opt_collapse(
				i * tranche,
				j * tranche,
				(i + 1) * tranche - 1,
				(j + 1) * tranche - 1
			);

			tab[i][j] = change;
		}
	}
	
	for (unsigned it = 1; it <= nb_iter; it++){

		change |= traiter_tuile_omp_tiled_opt_collapse(1, 1, tranche, DIM-2);
		change |= traiter_tuile_omp_tiled_opt_collapse(DIM-tranche, 1, DIM-2, DIM -2);
		change |= traiter_tuile_omp_tiled_opt_collapse(tranche+1, 1, DIM-tranche-1, tranche);
		change |= traiter_tuile_omp_tiled_opt_collapse(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

		for (int i = 1 ; i < GRAIN-1; i++){
			for (int j = 1 ; j < GRAIN-1; j++){
				
				if (!tab[i][j] && !tab[i-1][j] && !tab[i+1][j] && !tab[i][j-1] && !tab[i][j+1] && !tab[i+1][j+1] && !tab[i-1][j-1] && !tab[i+1][j-1] && !tab[i-1][j+1]){
					change |= traiter_tuile_omp_tiled_opt_collapse(
						i * tranche,
						j * tranche,
						(i + 1) * tranche - 1,
						(j + 1) * tranche - 1
					);

					tab[i][j] = change;
				}
			}
		}

		swap_images();

		if (!change)
		{
			return it;
		}
	}

	return 0;
}


// ============================== Version parallèle tuilée avec task ==============================

static int traiter_tuile_task_tiled(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}

unsigned vie_compute_task_tiled(unsigned nb_iter)
{

	tranche = DIM / GRAIN;
	unsigned change = 0;

	for (unsigned it = 1; it <= nb_iter; it++){

		for (int i = 0; i < GRAIN; i++){
			#pragma omp parallel
			#pragma omp single
			for (int j = 0; j < GRAIN; j++){
				#pragma omp task
				change |= traiter_tuile_seq_tiled(
					(i == 0) + i * tranche,
					(j == 0) + j * tranche,
					(i + 1) * tranche - 1 - (i == GRAIN-1),
					(j + 1) * tranche - 1 - (j == GRAIN-1));
			}
			#pragma omp taskwait
		}

		swap_images();

		if (!change)
			return it;
	}

	return 0;
}


// ============================== Version parallèle tuilée optimisée avec task ==============================

static int traiter_tuile_task_tiled_opt(int i_d, int j_d, int i_f, int j_f)
{
	unsigned change = 0;

	PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", i_d, i_f, j_d, j_f);

	for (int i = i_d; i <= i_f; i++)
		for (int j = j_d; j <= j_f; j++)
			change |= compute_new_state(i, j);
		

	return change;
}

unsigned vie_compute_task_tiled_opt(unsigned nb_iter)
{

	bool tab[GRAIN][GRAIN];
	for (int i = 0; i < GRAIN; i++)
		for (int j = 0 ; j < GRAIN ; j++)
			tab[i][j] = false;

	tranche = DIM / GRAIN;
	unsigned change = 0;

	change |= traiter_tuile_task_tiled_opt(1, 1, tranche, DIM-2);
	change |= traiter_tuile_task_tiled_opt(DIM-tranche, 1, DIM-2, DIM -2);
	change |= traiter_tuile_task_tiled_opt(tranche+1, 1, DIM-tranche-1, tranche);
	change |= traiter_tuile_task_tiled_opt(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

	for (int i = 1 ; i < GRAIN-1; i++){
		#pragma omp parallel
		#pragma omp single
		for (int j = 1 ; j < GRAIN-1; j++){
			#pragma omp task
			change |= traiter_tuile_task_tiled_opt(
				i * tranche,
				j * tranche,
				(i + 1) * tranche - 1,
				(j + 1) * tranche - 1
			);

			tab[i][j] = change;
		}
	}
	
	for (unsigned it = 1; it <= nb_iter; it++){

		change |= traiter_tuile_task_tiled_opt(1, 1, tranche, DIM-2);
		change |= traiter_tuile_task_tiled_opt(DIM-tranche, 1, DIM-2, DIM -2);
		change |= traiter_tuile_task_tiled_opt(tranche+1, 1, DIM-tranche-1, tranche);
		change |= traiter_tuile_task_tiled_opt(tranche+1, DIM-tranche, DIM-tranche-1, DIM -2);

		for (int i = 1 ; i < GRAIN-1; i++){
			#pragma omp parallel
			#pragma omp single
			for (int j = 1 ; j < GRAIN-1; j++){
				if (!tab[i][j] && !tab[i-1][j] && !tab[i+1][j] && !tab[i][j-1] && !tab[i][j+1] && !tab[i+1][j+1] && !tab[i-1][j-1] && !tab[i+1][j-1] && !tab[i-1][j+1]){
					#pragma omp task
					change |= traiter_tuile_task_tiled_opt(
						i * tranche,
						j * tranche,
						(i + 1) * tranche - 1,
						(j + 1) * tranche - 1
					);

					tab[i][j] = change;
				}
			}
			#pragma omp taskwait
		}

		swap_images();

		if (!change)
		{
			return it;
		}
	}

	return 0;
}


// ============================== Version OpenCL tuilée ==============================

unsigned vie_compute_ocl (unsigned nb_iter)
{

  size_t global[2] = {SIZE, SIZE};
  size_t local[2]  = {TILEX, TILEY};
  cl_int err;

  for (unsigned it = 1; it <= nb_iter; it++) {

    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
	err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local, 0, NULL, NULL);

    check (err, "Failed to execute kernel");
	
    cl_mem tmp  = cur_buffer;
    cur_buffer  = next_buffer;
    next_buffer = tmp;
	
  }

  return 0;
}


// ============================== Version MPI de base ==============================

#ifdef ENABLE_MPI

unsigned vie_compute_mpi_base(unsigned nb_iter)
{
	int mpi_rank, mpi_size, tranche;
	
  	MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

	MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

	tranche = DIM / mpi_size;

	for (unsigned it = 1; it <= nb_iter; it++){

		MPI_Scatter (&cur_img(0,0), tranche * DIM, MPI_UNSIGNED, &cur_img(0,0), tranche * DIM, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

		traiter_tuile_seq_tiled(1,1, tranche-2, DIM-2);

		MPI_Gather (&next_img(0,0), tranche * DIM, MPI_UNSIGNED, &next_img(0,0), tranche * DIM, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

		swap_images();
	}

	return 0;
}

#endif


// ============================== Version MPI + OpenMP ==============================

#ifdef ENABLE_MPI

unsigned vie_compute_mpi_omp(unsigned nb_iter)
{
	int mpi_rank, mpi_size, tranche;
	
  	MPI_Comm_rank (MPI_COMM_WORLD, &mpi_rank);

	MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);

	tranche = DIM / mpi_size;

	for (unsigned it = 1; it <= nb_iter; it++){

		MPI_Scatter (&cur_img(0,0), tranche * DIM, MPI_UNSIGNED, &cur_img(0,0), tranche * DIM, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

		traiter_tuile_omp_tiled_dynamic(1,1, tranche-2, DIM-2);

		MPI_Gather (&next_img(0,0), tranche * DIM, MPI_UNSIGNED, &next_img(0,0), tranche * DIM, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

		swap_images();
	}

	return 0;
}

#endif




///////////////////////////// Configuration initiale

void draw_stable(void);
void draw_guns(void);
void draw_random(void);
void draw_clown(void);
void draw_diehard(void);

void vie_draw(char *param)
{
	char func_name[1024];
	void (*f)(void) = NULL;

	if (param == NULL)
		f = draw_guns;
	else
	{
		sprintf(func_name, "draw_%s", param);
		f = dlsym(DLSYM_FLAG, func_name);

		if (f == NULL)
		{
			PRINT_DEBUG('g', "Cannot resolve draw function: %s\n", func_name);
			f = draw_guns;
		}
	}

	f();
}

static unsigned couleur = 0xFFFF00FF; // Yellow

static void gun(int x, int y, int version)
{
	bool glider_gun[11][38] = {
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
		 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
		 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0},
		{0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
		 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,
		 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
		 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	};

	if (version == 0)
		for (int i = 0; i < 11; i++)
			for (int j = 0; j < 38; j++)
				if (glider_gun[i][j])
					cur_img(i + x, j + y) = couleur;

	if (version == 1)
		for (int i = 0; i < 11; i++)
			for (int j = 0; j < 38; j++)
				if (glider_gun[i][j])
					cur_img(x - i, j + y) = couleur;

	if (version == 2)
		for (int i = 0; i < 11; i++)
			for (int j = 0; j < 38; j++)
				if (glider_gun[i][j])
					cur_img(x - i, y - j) = couleur;

	if (version == 3)
		for (int i = 0; i < 11; i++)
			for (int j = 0; j < 38; j++)
				if (glider_gun[i][j])
					cur_img(i + x, y - j) = couleur;
}

void draw_stable(void)
{
	for (int i = 1; i < DIM - 2; i += 4)
		for (int j = 1; j < DIM - 2; j += 4)
			cur_img(i, j) = cur_img(i, (j + 1)) = cur_img((i + 1), j) =
				cur_img((i + 1), (j + 1)) = couleur;
}

void draw_guns(void)
{
	memset(&cur_img(0, 0), 0, DIM * DIM * sizeof(cur_img(0, 0)));

	gun(0, 0, 0);
	gun(0, DIM - 1, 3);
	gun(DIM - 1, DIM - 1, 2);
	gun(DIM - 1, 0, 1);
}

void draw_random(void)
{
	for (int i = 1; i < DIM - 1; i++)
		for (int j = 1; j < DIM - 1; j++)
			cur_img(i, j) = random() & 01;
}

void draw_clown(void)
{
	memset(&cur_img(0, 0), 0, DIM * DIM * sizeof(cur_img(0, 0)));

	int mid = DIM / 2;
	cur_img(mid, mid - 1) = cur_img(mid, mid) = cur_img(mid, mid + 1) =
		couleur;
	cur_img(mid + 1, mid - 1) = cur_img(mid + 1, mid + 1) = couleur;
	cur_img(mid + 2, mid - 1) = cur_img(mid + 2, mid + 1) = couleur;
}

void draw_diehard(void)
{
	memset(&cur_img(0, 0), 0, DIM * DIM * sizeof(cur_img(0, 0)));

	int mid = DIM / 2;

	cur_img(mid, mid - 3) = cur_img(mid, mid - 2) = couleur;
	cur_img(mid + 1, mid - 2) = couleur;

	cur_img(mid - 1, mid + 3) = couleur;
	cur_img(mid + 1, mid + 2) = cur_img(mid + 1, mid + 3) =
		cur_img(mid + 1, mid + 4) = couleur;
}