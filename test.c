#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <pthread.h>
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define HEIGHT	1024
#define WIDTH	1024
#define NOISE_LIMIT	20

#define GREYSCALE '0'
#define SEPIA '1'
#define NOISE '2'
#define CLEAR '3'
#define BLUR '4'

typedef unsigned char byte;
typedef unsigned short word;
typedef unsigned int dword;

struct pixel{
	byte b;
	byte g;
	byte r;
}__attribute__((aligned(1)));

struct image{
	word width;
	word height;
	word num_colors;
	struct pixel *pixels;
}__attribute__((aligned(1)));

struct bmpHeader{
	byte type[2];
	byte reserved[16];
	word width;
	byte reserved2[2];
	word height;
	byte reserved3[22];
	word num_colors;
	byte reserved4[6];
}__attribute__((aligned(1)));

struct BMPIMG{
	struct bmpHeader * header;
	struct paletteInfo * palette;
	struct image * img;
}__attribute__((aligned(1)));

union{
	byte vect[54];
	struct bmpHeader h;
} hdr_t;

struct paletteInfo{

	byte *palette;
};
// attribute packed
char *filename;

typedef struct 
{
	word width;
	word height;
	struct pixel *pixels;
	long offset;
} TParam;

struct BMPIMG * deepCopy(struct BMPIMG *in)
{
	struct BMPIMG * out = (struct BMPIMG *)malloc(sizeof(struct BMPIMG));
	out->header = (struct bmpHeader *)malloc(sizeof(struct bmpHeader));
	memcpy(out->header, in->header, sizeof(struct bmpHeader));

	out->palette = (struct paletteInfo *)malloc(sizeof(struct paletteInfo));
	out->img = (struct image *)malloc(sizeof(struct image));
	
	out->palette->palette = (byte *)malloc(sizeof(byte)* 4* in->img->num_colors);
	memcpy(out->palette->palette, in->palette->palette, sizeof(byte)* 4 * in->img->num_colors);
	
	out->img->pixels = (struct pixel *)malloc(sizeof(struct pixel) * in->img->width * in->img->height);
	memcpy(out->img->pixels, in->img->pixels, sizeof(struct pixel) * in->img->width * in->img->height);
	out->img->width = in->img->width;
	out->img->height = in->img->height;
	out->img->num_colors = in->img->num_colors;

	return out;
}

void writeBMP(char *fname, struct BMPIMG * bmpimg)
{
	struct image *img = bmpimg->img;
	struct bmpHeader *header = bmpimg->header;
	struct paletteInfo* p = bmpimg->palette;
	
	FILE *f = fopen(fname, "wb+");
	byte *b = (byte *)header;
	int i = 0;
	for(i = 0; i < 54; i++)
		fputc(b[i], f);

	for(i = 0; i < 4 * img->num_colors; i++)
		fputc(p->palette[i], f);

	for(i = 0; i < img->width * img->height; i++)
	{	
		fputc(img->pixels[i].b, f);
		fputc(img->pixels[i].g, f);
		fputc(img->pixels[i].r, f);
	}
	fclose(f);
}

void grayscaleFilter(struct BMPIMG *bmpimg)
{
	int i;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	for(i = 0; i < width * height; i++)
	{
		byte red = bimg->pixels[i].r;
		byte blue = bimg->pixels[i].g;
		byte green = bimg->pixels[i].b;
		byte avg = (red/3 + blue/3 + green/3)%256;
		bimg->pixels[i].r = avg;
		bimg->pixels[i].g = avg;
		bimg->pixels[i].b = avg;
	}
}

void grayscaleFilterOMP(struct BMPIMG *bmpimg)
{		
	#pragma omp parallel
	{	
		struct image *bimg = bmpimg->img;
		int width = bimg->width;
		int height = bimg->height;
		struct pixel *p;
		struct pixel *end = bimg->pixels + width * height;
		
		#pragma omp for
		for (p = bimg->pixels; p < end; p++) {
			byte red = p->r;
			byte blue = p->b;
			byte green = p->g;
			byte avg = (red/3 + blue/3 + green/3)%256;
			p->r = avg;
			p->g = avg;
			p->b = avg;
		}
	}
}

void* grayscaleTask(void* params) {
	int i;
	TParam *bimg = (TParam*)params;
	int width = bimg->width;
	int height = bimg->height;
	for(i = 0; i < width * height; i++)
	{
		byte red = bimg->pixels[i].r;
		byte blue = bimg->pixels[i].g;
		byte green = bimg->pixels[i].b;
		byte avg = (red/3 + blue/3 + green/3)%256;
		bimg->pixels[i].r = avg;
		bimg->pixels[i].g = avg;
		bimg->pixels[i].b = avg;
	}
	return NULL;
}

void runOnThreads(struct BMPIMG *bmpimg, int num_threads, void* (*task)(void*)) {
	int i;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	int chunk = height/num_threads;
	pthread_t threads[num_threads];
	TParam p[num_threads];
	long offset = 0;
	int rem_height = height;

	for (i = 0; i < num_threads-1; i++) {
		p[i].width = width;
		p[i].height = chunk;
		p[i].pixels = (struct pixel*)malloc(width*chunk*sizeof(struct pixel));
		p[i].offset = offset;
		memcpy(p[i].pixels, bimg->pixels+offset, width*chunk*sizeof(struct pixel));
		offset += width*chunk;
		rem_height -= chunk;
		if (pthread_create(&threads[i], NULL, task, &p[i]))
            perror("pthread_create");
	}

	p[i].width = width;
	p[i].height = rem_height;
	p[i].pixels = (struct pixel*)malloc(width*rem_height*sizeof(struct pixel));
	p[i].offset = offset;
	memcpy(p[i].pixels, bimg->pixels+offset, width*rem_height*sizeof(struct pixel));
	if (pthread_create(&threads[i], NULL, task, &p[i]))
        perror("pthread_create");

	for (i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL))
            perror("pthread_join");
    }

    offset = 0;
    for (i = 0; i < num_threads; i++) {
    	memcpy(bimg->pixels+offset, p[i].pixels, p[i].width*p[i].height*sizeof(struct pixel));
    	offset += p[i].width*p[i].height;
    }
}

void runOnThreadsExtendMargins(struct BMPIMG *bmpimg, int num_threads, void* (*task)(void*)) {
	int i;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	int chunk = height/num_threads;
	pthread_t threads[num_threads];
	TParam p[num_threads];
	long offset = 0;
	int rem_height = height;

	p[0].width = width;
	if (num_threads > 1){
		p[0].height = chunk + 1;
		p[0].pixels = (struct pixel*)malloc(width*(chunk+1)*sizeof(struct pixel));
		p[0].offset = offset;
		memcpy(p[0].pixels, bimg->pixels, width*(chunk+1)*sizeof(struct pixel));
		offset += width*(chunk-1);
		rem_height -= (chunk-1);
	}
	else {
		p[0].height = chunk;
		p[0].pixels = (struct pixel*)malloc(width*chunk*sizeof(struct pixel));
		p[0].offset = offset;
		memcpy(p[0].pixels, bimg->pixels, width*chunk*sizeof(struct pixel));
		offset += width*chunk;
		rem_height -= chunk;
	}

	for (i = 1; i < num_threads-1; i++) {
		p[i].width = width;
		p[i].height = chunk + 2;
		p[i].pixels = (struct pixel*)malloc(width*(chunk+2)*sizeof(struct pixel));
		p[i].offset = offset;
		memcpy(p[i].pixels, bimg->pixels+offset, width*(chunk+2)*sizeof(struct pixel));
		offset += width*chunk;
		rem_height -= chunk;
	}

	if (num_threads > 1) {
		p[i].width = width;
		p[i].height = rem_height;
		p[i].pixels = (struct pixel*)malloc(width*rem_height*sizeof(struct pixel));
		memcpy(p[i].pixels, bimg->pixels+offset, width*rem_height*sizeof(struct pixel));
	}

	for (i = 0; i < num_threads; i++) {
		if (pthread_create(&threads[i], NULL, task, &p[i]))
            perror("pthread_create");
	}

	for (i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL))
            perror("pthread_join");
    }

    offset = 0;
    if (num_threads > 1) {
    	memcpy(bimg->pixels, p[0].pixels, p[0].width*(p[0].height-1)*sizeof(struct pixel));
    	offset += p[0].width*(p[0].height-1);
    }
    else 
    	memcpy(bimg->pixels, p[0].pixels, p[0].width*p[0].height*sizeof(struct pixel));

    for (i = 1; i < num_threads-1; i++) {
    	memcpy(bimg->pixels+offset, p[i].pixels+p[i].width, p[i].width*(p[i].height-2)*sizeof(struct pixel));
    	offset += p[i].width*(p[i].height-2);
    }

    if (num_threads > 1)
    	memcpy(bimg->pixels+offset, p[i].pixels+p[i].width, p[i].width*(p[i].height-1)*sizeof(struct pixel));
}

void sepiaFilter(struct BMPIMG *bmpimg)
{
	int i;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	for(i = 0; i < width * height; i++)
	{
		byte red = bimg->pixels[i].r;
		byte blue = bimg->pixels[i].g;
		byte green = bimg->pixels[i].b;

		byte outRed = (byte)MIN((red * 0.393f + green * 0.769f + blue * 0.189f), 255);
		byte outGreen = (byte)MIN((red * 0.349f + green * 0.686f + blue * 0.168f), 255);
		byte outBlue = (byte)MIN((red * 0.272f + green * 0.534f + blue * 0.131f), 255);
		
		bimg->pixels[i].r = outRed;
		bimg->pixels[i].g = outGreen;
		bimg->pixels[i].b = outBlue;
	}	
}

void sepiaFilterOMP(struct BMPIMG *bmpimg)
{
	#pragma omp parallel
	{	
		struct image *bimg = bmpimg->img;
		int width = bimg->width;
		int height = bimg->height;
		struct pixel *p;
		struct pixel *end = bimg->pixels + width * height;
		#pragma omp for
		for (p = bimg->pixels; p < end; p++) {

			byte red = p->r;
			byte blue = p->g;
			byte green = p->b;

			byte outRed = (byte)MIN((red * 0.393f + green * 0.769f + blue * 0.189f), 255);
			byte outGreen = (byte)MIN((red * 0.349f + green * 0.686f + blue * 0.168f), 255);
			byte outBlue = (byte)MIN((red * 0.272f + green * 0.534f + blue * 0.131f), 255);
			
			p->r = outRed;
			p->g = outGreen;
			p->b = outBlue;
		}
	}	
}

void* sepiaTask(void* params) {
	int i;
	TParam *bimg = (TParam *)params;
	int width = bimg->width;
	int height = bimg->height;
	for(i = 0; i < width * height; i++)
	{
		byte red = bimg->pixels[i].r;
		byte blue = bimg->pixels[i].g;
		byte green = bimg->pixels[i].b;

		byte outRed = (byte)MIN((red * 0.393f + green * 0.769f + blue * 0.189f), 255);
		byte outGreen = (byte)MIN((red * 0.349f + green * 0.686f + blue * 0.168f), 255);
		byte outBlue = (byte)MIN((red * 0.272f + green * 0.534f + blue * 0.131f), 255);
		
		bimg->pixels[i].r = outRed;
		bimg->pixels[i].g = outGreen;
		bimg->pixels[i].b = outBlue;
	}
	return NULL;
}

void insertNoise(struct BMPIMG *bmpimg)
{
	int i;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	for(i = 0; i < width * height; i++)
	{
		int ii = i / width;
		int jj = i % width;
		if( ii % 5 == 0 && jj %5 == 0)	
			bimg->pixels[i].r = bimg->pixels[i].g = bimg->pixels[i].b = ((int)bimg->pixels[i].r/129)*255;
	}
}

void insertNoiseOMP(struct BMPIMG *bmpimg)
{
	#pragma omp parallel
	{	
		int i;
		struct image *bimg = bmpimg->img;
		int width = bimg->width;
		int height = bimg->height;

		#pragma omp for
		for(i = 0; i < width * height; i++)
		{
			int ii = i / width;
			int jj = i % width;
			if( ii % 5 == 0 && jj %5 == 0)	
				bimg->pixels[i].r = bimg->pixels[i].g = bimg->pixels[i].b = ((int)bimg->pixels[i].r/129)*255;
		}
	}
}

void* noiseTask(void* params) {
	int i;
	TParam *bimg = (TParam*)params;
	int width = bimg->width;
	int height = bimg->height;
	long offset = bimg->offset;
	for(i = 0; i < width * height; i++)
	{
		long ii = (i + offset) / width;
		long jj = (i + offset) % width;
		if( ii % 5 == 0 && jj % 5 == 0)	
			bimg->pixels[i].r = bimg->pixels[i].g = bimg->pixels[i].b = ((int)bimg->pixels[i].r/129)*255;
	}
	return NULL;
}

/* transforms each pixel to grayscale and and
replaces all r g b components with mean value
if it's needed */
void medianFilter(struct BMPIMG *bmpimg)
{
	int i;
	int j;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	//printf("%d x %d:%d %d\n", width, height, width*height, strlen(bimg->pixels));
	int meanR, meanG, meanB, grayMean, current_pixel_mean;
	for(i = 1 ; i < height-1; i++)
		for(j = 1; j < width-1; j++)
		{
			meanR = bimg->pixels[ i*width + j + 1].r + bimg->pixels[ i*width + j - 1].r +
					bimg->pixels[ (i+1)*width + j].r + bimg->pixels[ (i+1)*width + j + 1].r + bimg->pixels[ (i+1)*width + j - 1].r +
					bimg->pixels[ (i-1)*width + j].r + bimg->pixels[ (i-1)*width + j + 1].r + bimg->pixels[ (i-1)*width + j - 1].r;  
			meanR /=8;

			meanG = bimg->pixels[ i*width + j + 1].g + bimg->pixels[ i*width + j - 1].g +
					bimg->pixels[ (i+1)*width + j].g + bimg->pixels[ (i+1)*width + j + 1].g + bimg->pixels[ (i+1)*width + j - 1].g +
					bimg->pixels[ (i-1)*width + j].g + bimg->pixels[ (i-1)*width + j + 1].g + bimg->pixels[ (i-1)*width + j - 1].g;  
			meanG /=8;
			
			meanB = bimg->pixels[ i*width + j + 1].b + bimg->pixels[ i*width + j - 1].b +
					bimg->pixels[ (i+1)*width + j].b + bimg->pixels[ (i+1)*width + j + 1].b + bimg->pixels[ (i+1)*width + j - 1].b +
					bimg->pixels[ (i-1)*width + j].b + bimg->pixels[ (i-1)*width + j + 1].b + bimg->pixels[ (i-1)*width + j - 1].b;  
			meanB /=8;

			grayMean =(meanB + meanG + meanR)/3;
			current_pixel_mean = (bimg->pixels[i*width + j].r + bimg->pixels[i*width + j].g + bimg->pixels[i*width + j].b)/3;
			if (current_pixel_mean < grayMean - NOISE_LIMIT || current_pixel_mean > grayMean + NOISE_LIMIT)
			{	
				bimg->pixels[i*width + j].r = meanR;
				bimg->pixels[i*width + j].g = meanG;
				bimg->pixels[i*width + j].b = meanB;
			}

		}
}

void medianFilterOMP(struct BMPIMG *bmpimg)
{
	struct BMPIMG *bmpcopy = deepCopy(bmpimg);
	#pragma omp parallel
	{
		int i;
		int j;
		struct image *bimg = bmpcopy->img;
		struct image *bimgres = bmpimg->img;
		int width = bimg->width;
		int height = bimg->height;
		//printf("%d x %d:%d %d\n", width, height, width*height, strlen(bimg->pixels));
		int meanR, meanG, meanB, grayMean, current_pixel_mean;

		#pragma omp for collapse(2)
		for(i = 1 ; i < height-1; i++)
			for(j = 1; j < width-1; j++)
			{
				meanR = bimg->pixels[ i*width + j + 1].r + bimg->pixels[ i*width + j - 1].r +
						bimg->pixels[ (i+1)*width + j].r + bimg->pixels[ (i+1)*width + j + 1].r + bimg->pixels[ (i+1)*width + j - 1].r +
						bimg->pixels[ (i-1)*width + j].r + bimg->pixels[ (i-1)*width + j + 1].r + bimg->pixels[ (i-1)*width + j - 1].r;  
				meanR /=8;

				meanG = bimg->pixels[ i*width + j + 1].g + bimg->pixels[ i*width + j - 1].g +
						bimg->pixels[ (i+1)*width + j].g + bimg->pixels[ (i+1)*width + j + 1].g + bimg->pixels[ (i+1)*width + j - 1].g +
						bimg->pixels[ (i-1)*width + j].g + bimg->pixels[ (i-1)*width + j + 1].g + bimg->pixels[ (i-1)*width + j - 1].g;  
				meanG /=8;
				
				meanB = bimg->pixels[ i*width + j + 1].b + bimg->pixels[ i*width + j - 1].b +
						bimg->pixels[ (i+1)*width + j].b + bimg->pixels[ (i+1)*width + j + 1].b + bimg->pixels[ (i+1)*width + j - 1].b +
						bimg->pixels[ (i-1)*width + j].b + bimg->pixels[ (i-1)*width + j + 1].b + bimg->pixels[ (i-1)*width + j - 1].b;  
				meanB /=8;

				grayMean =(meanB + meanG + meanR)/3;
				current_pixel_mean = (bimg->pixels[i*width + j].r + bimg->pixels[i*width + j].g + bimg->pixels[i*width + j].b)/3;
				if (current_pixel_mean < grayMean - NOISE_LIMIT || current_pixel_mean > grayMean + NOISE_LIMIT)
				{	
					bimgres->pixels[i*width + j].r = meanR;
					bimgres->pixels[i*width + j].g = meanG;
					bimgres->pixels[i*width + j].b = meanB;
				}
			}
	}
}

void* medianFilterTask(void* params) {
	int i;
	int j;
	TParam *bimg;
	TParam *bimgres = (TParam*)params;
	int width = bimgres->width;
	int height = bimgres->height;
	//printf("%d x %d:%d %d\n", width, height, width*height, strlen(bimg->pixels));
	int meanR, meanG, meanB, grayMean, current_pixel_mean;

	bimg = (TParam*)malloc(sizeof(TParam));

	bimg->pixels = (struct pixel*)malloc(width*height*sizeof(struct pixel));
	memcpy(bimg->pixels, bimgres->pixels, width*height*sizeof(struct pixel));

	for(i = 1 ; i < height-1; i++)
		for(j = 1; j < width-1; j++)
		{
			meanR = bimg->pixels[ i*width + j + 1].r + bimg->pixels[ i*width + j - 1].r +
					bimg->pixels[ (i+1)*width + j].r + bimg->pixels[ (i+1)*width + j + 1].r + bimg->pixels[ (i+1)*width + j - 1].r +
					bimg->pixels[ (i-1)*width + j].r + bimg->pixels[ (i-1)*width + j + 1].r + bimg->pixels[ (i-1)*width + j - 1].r;  
			meanR /=8;

			meanG = bimg->pixels[ i*width + j + 1].g + bimg->pixels[ i*width + j - 1].g +
					bimg->pixels[ (i+1)*width + j].g + bimg->pixels[ (i+1)*width + j + 1].g + bimg->pixels[ (i+1)*width + j - 1].g +
					bimg->pixels[ (i-1)*width + j].g + bimg->pixels[ (i-1)*width + j + 1].g + bimg->pixels[ (i-1)*width + j - 1].g;  
			meanG /=8;
			
			meanB = bimg->pixels[ i*width + j + 1].b + bimg->pixels[ i*width + j - 1].b +
					bimg->pixels[ (i+1)*width + j].b + bimg->pixels[ (i+1)*width + j + 1].b + bimg->pixels[ (i+1)*width + j - 1].b +
					bimg->pixels[ (i-1)*width + j].b + bimg->pixels[ (i-1)*width + j + 1].b + bimg->pixels[ (i-1)*width + j - 1].b;  
			meanB /=8;

			grayMean =(meanB + meanG + meanR)/3;
			current_pixel_mean = (bimg->pixels[i*width + j].r + bimg->pixels[i*width + j].g + bimg->pixels[i*width + j].b)/3;
			if (current_pixel_mean < grayMean - NOISE_LIMIT || current_pixel_mean > grayMean + NOISE_LIMIT)
			{	
				bimgres->pixels[i*width + j].r = meanR;
				bimgres->pixels[i*width + j].g = meanG;
				bimgres->pixels[i*width + j].b = meanB;
			}

		}
	return NULL;
}

/* sets each r g b byte to it's mean value and
and makes checks if exceds NOISE_LIMIT 
LE: NOT WORKING AS INTENDED
*/
void medianFilterRGB(struct BMPIMG *bmpimg)
{
	int i;
	int j;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	//printf("%d x %d:%d %d\n", width, height, width*height, strlen(bimg->pixels));
	int meanR, meanG, meanB;
	for(i = 1 ; i < height-1; i++)
		for(j = 1; j < width-1; j++)
		{
			meanR = bimg->pixels[ i*width + j + 1].r + bimg->pixels[ i*width + j - 1].r +
					bimg->pixels[ (i+1)*width + j].r + bimg->pixels[ (i+1)*width + j + 1].r + bimg->pixels[ (i+1)*width + j - 1].r +
					bimg->pixels[ (i-1)*width + j].r + bimg->pixels[ (i-1)*width + j + 1].r + bimg->pixels[ (i-1)*width + j - 1].r;  
			meanR /=8;

			meanG = bimg->pixels[ i*width + j + 1].g + bimg->pixels[ i*width + j - 1].g +
					bimg->pixels[ (i+1)*width + j].g + bimg->pixels[ (i+1)*width + j + 1].g + bimg->pixels[ (i+1)*width + j - 1].g +
					bimg->pixels[ (i-1)*width + j].g + bimg->pixels[ (i-1)*width + j + 1].g + bimg->pixels[ (i-1)*width + j - 1].g;  
			meanG /=8;
			
			meanB = bimg->pixels[ i*width + j + 1].b + bimg->pixels[ i*width + j - 1].b +
					bimg->pixels[ (i+1)*width + j].b + bimg->pixels[ (i+1)*width + j + 1].b + bimg->pixels[ (i+1)*width + j - 1].b +
					bimg->pixels[ (i-1)*width + j].b + bimg->pixels[ (i-1)*width + j + 1].b + bimg->pixels[ (i-1)*width + j - 1].b;  
			meanB /=8;

			if (bimg->pixels[i*width + j].r < meanR - NOISE_LIMIT || bimg->pixels[i*width + j].r > meanR + NOISE_LIMIT)
				bimg->pixels[i*width + j].r = meanR;
			
			if (bimg->pixels[i*width + j].g < meanG - NOISE_LIMIT || bimg->pixels[i*width + j].g > meanG + NOISE_LIMIT)
				bimg->pixels[i*width + j].g = meanG;
			
			if (bimg->pixels[i*width + j].g < meanB - NOISE_LIMIT || bimg->pixels[i*width + j].g > meanB + NOISE_LIMIT)
				bimg->pixels[i*width + j].g = meanB;
		}
}


void blur3x3(struct BMPIMG *bmpimg)
{
	int i;
	int j;
	struct image *bimg = bmpimg->img;
	int width = bimg->width;
	int height = bimg->height;
	//printf("%d x %d:%d %d\n", width, height, width*height, strlen(bimg->pixels));
	int meanR, meanG, meanB;
	for(i = 1 ; i < height-1; i++)
		for(j = 1; j < width-1; j++)
		{
			meanR = bimg->pixels[i*width + j].r + bimg->pixels[ i*width + j + 1].r + bimg->pixels[ i*width + j - 1].r +
					bimg->pixels[ (i+1)*width + j].r + bimg->pixels[ (i+1)*width + j + 1].r + bimg->pixels[ (i+1)*width + j - 1].r +
					bimg->pixels[ (i-1)*width + j].r + bimg->pixels[ (i-1)*width + j + 1].r + bimg->pixels[ (i-1)*width + j - 1].r;  
			meanR /=9;

			meanG = bimg->pixels[i*width + j].g + bimg->pixels[ i*width + j + 1].g + bimg->pixels[ i*width + j - 1].g +
					bimg->pixels[ (i+1)*width + j].g + bimg->pixels[ (i+1)*width + j + 1].g + bimg->pixels[ (i+1)*width + j - 1].g +
					bimg->pixels[ (i-1)*width + j].g + bimg->pixels[ (i-1)*width + j + 1].g + bimg->pixels[ (i-1)*width + j - 1].g;  
			meanG /=9;
			
			meanB = bimg->pixels[i*width + j].b + bimg->pixels[ i*width + j + 1].b + bimg->pixels[ i*width + j - 1].b +
					bimg->pixels[ (i+1)*width + j].b + bimg->pixels[ (i+1)*width + j + 1].b + bimg->pixels[ (i+1)*width + j - 1].b +
					bimg->pixels[ (i-1)*width + j].b + bimg->pixels[ (i-1)*width + j + 1].b + bimg->pixels[ (i-1)*width + j - 1].b;  
			meanB /=9;

			bimg->pixels[i*width + j].r = meanR;
			bimg->pixels[i*width + j].g = meanG;
			bimg->pixels[i*width + j].b = meanB;
		}
}

void blur3x3OMP(struct BMPIMG *bmpimg)
{
	struct BMPIMG *bmpcopy = deepCopy(bmpimg);
	#pragma omp parallel
	{
		int i;
		int j;
		struct image *bimg = bmpcopy->img;
		struct image *bimgres = bmpimg->img;
		int width = bimg->width;
		int height = bimg->height;
		//printf("%d x %d:%d %d\n", width, height, width*height, strlen(bimg->pixels));
		int meanR, meanG, meanB;

		#pragma omp for collapse(2)
		for(i = 1 ; i < height-1; i++)
			for(j = 1; j < width-1; j++)
			{
				meanR = bimg->pixels[i*width + j].r + bimg->pixels[ i*width + j + 1].r + bimg->pixels[ i*width + j - 1].r +
						bimg->pixels[ (i+1)*width + j].r + bimg->pixels[ (i+1)*width + j + 1].r + bimg->pixels[ (i+1)*width + j - 1].r +
						bimg->pixels[ (i-1)*width + j].r + bimg->pixels[ (i-1)*width + j + 1].r + bimg->pixels[ (i-1)*width + j - 1].r;  
				meanR /=9;

				meanG = bimg->pixels[i*width + j].g + bimg->pixels[ i*width + j + 1].g + bimg->pixels[ i*width + j - 1].g +
						bimg->pixels[ (i+1)*width + j].g + bimg->pixels[ (i+1)*width + j + 1].g + bimg->pixels[ (i+1)*width + j - 1].g +
						bimg->pixels[ (i-1)*width + j].g + bimg->pixels[ (i-1)*width + j + 1].g + bimg->pixels[ (i-1)*width + j - 1].g;  
				meanG /=9;
				
				meanB = bimg->pixels[i*width + j].b + bimg->pixels[ i*width + j + 1].b + bimg->pixels[ i*width + j - 1].b +
						bimg->pixels[ (i+1)*width + j].b + bimg->pixels[ (i+1)*width + j + 1].b + bimg->pixels[ (i+1)*width + j - 1].b +
						bimg->pixels[ (i-1)*width + j].b + bimg->pixels[ (i-1)*width + j + 1].b + bimg->pixels[ (i-1)*width + j - 1].b;  
				meanB /=9;

				bimgres->pixels[i*width + j].r = meanR;
				bimgres->pixels[i*width + j].g = meanG;
				bimgres->pixels[i*width + j].b = meanB;
			}
	}
}

void* blurTask(void* params) {
	int i;
	int j;
	TParam *bimg;
	TParam *bimgres = (TParam*)params;
	int width = bimgres->width;
	int height = bimgres->height;
	//printf("%d x %d:%d %d\n", width, height, width*height, strlen(bimg->pixels));
	int meanR, meanG, meanB;

	bimg = (TParam*)malloc(sizeof(TParam));

	bimg->pixels = (struct pixel*)malloc(width*height*sizeof(struct pixel));
	memcpy(bimg->pixels, bimgres->pixels, width*height*sizeof(struct pixel));

	for(i = 1 ; i < height-1; i++)
		for(j = 1; j < width-1; j++)
		{
			meanR = bimg->pixels[i*width + j].r + bimg->pixels[ i*width + j + 1].r + bimg->pixels[ i*width + j - 1].r +
					bimg->pixels[ (i+1)*width + j].r + bimg->pixels[ (i+1)*width + j + 1].r + bimg->pixels[ (i+1)*width + j - 1].r +
					bimg->pixels[ (i-1)*width + j].r + bimg->pixels[ (i-1)*width + j + 1].r + bimg->pixels[ (i-1)*width + j - 1].r;  
			meanR /=9;

			meanG = bimg->pixels[i*width + j].g + bimg->pixels[ i*width + j + 1].g + bimg->pixels[ i*width + j - 1].g +
					bimg->pixels[ (i+1)*width + j].g + bimg->pixels[ (i+1)*width + j + 1].g + bimg->pixels[ (i+1)*width + j - 1].g +
					bimg->pixels[ (i-1)*width + j].g + bimg->pixels[ (i-1)*width + j + 1].g + bimg->pixels[ (i-1)*width + j - 1].g;  
			meanG /=9;
			
			meanB = bimg->pixels[i*width + j].b + bimg->pixels[ i*width + j + 1].b + bimg->pixels[ i*width + j - 1].b +
					bimg->pixels[ (i+1)*width + j].b + bimg->pixels[ (i+1)*width + j + 1].b + bimg->pixels[ (i+1)*width + j - 1].b +
					bimg->pixels[ (i-1)*width + j].b + bimg->pixels[ (i-1)*width + j + 1].b + bimg->pixels[ (i-1)*width + j - 1].b;  
			meanB /=9;

			bimgres->pixels[i*width + j].r = meanR;
			bimgres->pixels[i*width + j].g = meanG;
			bimgres->pixels[i*width + j].b = meanB;
		}
	return NULL;
}

/* read whole image */
struct BMPIMG* readBMP(char *fname)
{
	FILE *f = fopen(fname, "rb");
	struct image *bimg = (struct image*)malloc(sizeof(struct image));

	if(f == NULL || bimg == NULL)
	{
		printf("Error on alloc or fopen.\n");
		return NULL;
	}
	struct bmpHeader *header = (struct bmpHeader *)malloc(sizeof(struct bmpHeader));
	
	fread(header, sizeof(struct bmpHeader), 1, f);

	if(header->type[0] != 'B' || header->type[1] != 'M')
	{
		printf("File %s is not a bitmap image.\n", fname);
		return NULL;
	}

	bimg->width = header->width;
	bimg->height = header->height;
	bimg->num_colors = header->num_colors;
	
	//if (bimg->num_colors == 0)
	//	bimg->num_colors = 256;
	
	printf("%d x %d , Palette colors:%d\n", bimg->width, bimg->height, bimg->num_colors );
	
	struct paletteInfo p;
	p.palette = (byte *)malloc(4*bimg->num_colors);
	fread(p.palette, 4 * bimg->num_colors, 1, f);
	
	//printf("%d\n",*(dword *)&header->reserved[8] );
	
	bimg->pixels = (struct pixel *)malloc(sizeof(struct pixel) * bimg->width * bimg->height);
	if(bimg->pixels == NULL)
	{
		printf("Error when tryed to alloc buffer\n");
		return NULL;
	}

	fread(bimg->pixels, sizeof(struct pixel), bimg->width * bimg->height, f);
	
	struct BMPIMG *bmpimg = (struct BMPIMG *)malloc(sizeof(struct BMPIMG));
	bmpimg->header = header;
	bmpimg->palette = &p;
	bmpimg->img = bimg;

	return bmpimg;
}

int main(int argc, char *argv[])
{
	int i;
	if(argc != 2 && argc != 3)
	{	
		printf("Image name is missing. Send image name as parameter [ BLUR_LEVEL ]. Exiting...\n");
		return -1;
	}
	struct BMPIMG* bmpimg;
	printf("%s\n", argv[1]);
	filename = argv[1];
	struct timeval t1, t2;
    double elapsedTime;
	
	char outimg[50];
	int BLUR_LEVEL = 1;
	
	if(argc == 3)
		BLUR_LEVEL =  atoi(argv[2]);

	bmpimg = readBMP(filename);
/*
	sprintf(outimg,"original_%s", filename);
	writeBMP(outimg, bmpimg);
*/	
//	struct BMPIMG *sepiaImg = deepCopy(bmpimg);
//	struct BMPIMG *grayscaleImg = deepCopy(bmpimg);
	struct BMPIMG *noiseImg = deepCopy(bmpimg);
//	struct BMPIMG *blur3x3Img = deepCopy(bmpimg);

/*	gettimeofday(&t1, NULL); 
	//grayscaleFilter(grayscaleImg);
	//grayscaleFilterOMP(grayscaleImg);
	//runOnThreads(grayscaleImg, 2, grayscaleTask);
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf ("TIME GreyScale = %lf\n", elapsedTime);  
	sprintf(outimg,"grayscale_%s", filename);
	writeBMP(outimg, grayscaleImg);


	gettimeofday(&t1, NULL);
	//sepiaFilter(sepiaImg);
	//sepiaFilterOMP(sepiaImg);
	runOnThreads(sepiaImg, 4, sepiaTask);
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf ("TIME Sepia = %lf\n", elapsedTime);  
	sprintf(outimg,"sepia_%s", filename);
	writeBMP(outimg, sepiaImg);
*/
	gettimeofday(&t1, NULL); 
	//insertNoise(noiseImg);
	//insertNoiseOMP(noiseImg);
	runOnThreads(noiseImg, 1, noiseTask);
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf ("TIME Noise = %lf\n", elapsedTime);
	sprintf(outimg,"noise_%s", filename);  
	writeBMP(outimg, noiseImg);
/*
	struct BMPIMG *noiseFreeImg = deepCopy(noiseImg);

	gettimeofday(&t1, NULL);
	//medianFilter(noiseFreeImg);
	//medianFilterOMP(noiseFreeImg);
	runOnThreadsExtendMargins(noiseFreeImg, 1, medianFilterTask);
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf ("TIME Cleaned = %lf\n", elapsedTime);  
	sprintf(outimg,"cleaned_%s", filename);
	writeBMP(outimg, noiseFreeImg);

	gettimeofday(&t1, NULL);
	//for(i = 0; i < BLUR_LEVEL; i++)
	//	blur3x3(blur3x3Img);
	//for(i = 0; i < BLUR_LEVEL; i++)
	//	blur3x3OMP(blur3x3Img);
	for(i = 0; i < BLUR_LEVEL; i++)
		runOnThreadsExtendMargins(blur3x3Img, 1, blurTask);
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf ("TIME Blur3x3 = %lf\n", elapsedTime);  

	sprintf(outimg,"blur3x3_%s", filename);
	writeBMP(outimg, blur3x3Img);
*/
	return 0;
}