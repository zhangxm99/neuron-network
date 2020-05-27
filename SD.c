#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define InNeuron 2
#define OutNeuron 1
#define DataNum 820
#define Width 70
#define LearningRate 0.000001
#define TrainTime 10000
#define e 0.01

double InputMatrix[DataNum][InNeuron + 1] , OutputMatrix[DataNum][OutNeuron];
double WeightMatrix1[InNeuron + 1][Width] , WeightMatrix2[Width][OutNeuron];
double Maxin[InNeuron] , Minin[InNeuron] , Maxout[OutNeuron] , Minout[OutNeuron];
double ActivedHiddenLayer[Width];
double Gradient2[Width], Gradient1[InNeuron + 1][Width];


void ReadIn(void);
void InitialNetwork(void);
double Compute(int num);
void Train(void);
void writeNeuron(void);
double result(double var1,double var2);

int main(void){
    puts("  _   _  \n"                                
    "| \\ | | ___ _   _ _ __ ___  _ __ \n"       
    "|  \\| |/ _ \\ | | | '__/ _ \\| '_ \\   \n"    
    "| |\\  |  __/ |_| | | | (_) | | | |     \n" 
    "|_| \\_|\\___|\\__,_|_|  \\___/|_| |_| _    \n"
    "| \\ | | ___| |___      _____  _ __| | __\n"
    "|  \\| |/ _ \\ __\\ \\ /\\ / / _ \\| '__| |/ /\n"
    "| |\\  |  __/ |_ \\ V  V / (_) | |  |   < \n"
    "|_| \\_|\\___|\\__| \\_/\\_/ \\___/|_|  |_|\\_\\\n");

    ReadIn();
    InitialNetwork();
    Train();
    writeNeuron();
    printf("%lf\n",result(4.0,2.0));




    return 0;
}

void ReadIn(){
    FILE *fin ,*fot;

    if((fin = fopen("input.txt","r")) == NULL || (fot = fopen("output.txt","r") )== NULL){
        fputs("No input data\n",stderr);
        exit(0);
    }

    for(int i = 0;i < DataNum;i++){
        for(int j1 = 1;j1 < InNeuron + 1;j1++){
            fscanf(fin,"%lf",&InputMatrix[i][j1]);
        }
        for(int j2 = 0;j2 < OutNeuron;j2++){
            fscanf(fot,"lf",&OutputMatrix[i][j2]);
        }
    }
    fclose(fin);
    fclose(fot);
}

void InitialNetwork(){
    for(int j = 1; j < InNeuron + 1;j++){
        Maxin[j] = Minin[j] = InputMatrix[0][j];
        for(int i = 0;i < DataNum;i++){
            Maxin[j] = Maxin[j] > InputMatrix[i][j] ? Maxin[j] : InputMatrix[i][j];
            Maxin[j] = Minin[j] > InputMatrix[i][j] ? Minin[j] : InputMatrix[i][j];
        }
    }

    for(int j = 0;j < OutNeuron;j++){
        Maxout[j] = Minout[j] = OutputMatrix[0][j];
        for(int i = 0;i < DataNum;i++){
            Maxout[j] = Maxout[j] > OutputMatrix[i][j] ? Maxout[j] : OutputMatrix[i][j];
            Maxout[j] = Minin[j] > OutputMatrix[i][j] ? Minout[j] : OutputMatrix[i][j];
        }
    }

    for(int j = 1;j < InNeuron + 1;j++){
        for(int i = 0;i < DataNum;i++){
            InputMatrix[i][j] = (InputMatrix[i][j] - Minin[j] + 1) / (Maxin[j] - Maxin[j] + 1);
            InputMatrix[i][0] = 1;
        }
    }

    for(int j = 0;j < OutNeuron;j++){
        for(int i = 0;i < DataNum;i++){
            OutputMatrix[i][j] = (OutputMatrix[i][j] - Minout[j] + 1) / (Maxout[j] - Maxout[j] + 1);
        }
    }

    for(int i = 0;i < InNeuron + 1;i++){
        for(int j = 0;j < Width;j++){
            WeightMatrix1[i][j] = rand()*2.0/RAND_MAX-1;
        }
    }
    for(int i = 0;i < Width;i++){
        for(int j = 0;j < OutNeuron;j++){
            WeightMatrix2[i][j] = rand()*2.0/RAND_MAX-1;
        }
    }
}

double Compute(int num){
    for(int i = 0;i < Width;i++){
        double sum = 0;
        for(int j = 0;j < InNeuron + 1;j++){
            sum += InputMatrix[num][j] * WeightMatrix1[i][j];
        }
        sum = 1/(1+exp(-1*sum));
        ActivedHiddenLayer[i] = sum;
    }
    double y = 0;
    for(int i = 0;i < Width;i++){
        y += ActivedHiddenLayer[i] * WeightMatrix2[i][0];
    }
    return y;
}

void Train(){
    double Lost;
    int Train = 0;
    do{
        Train++;
        for(int i = 0;i < DataNum;i++){
            double y = Compute(i);
            double gap = y - OutputMatrix[0][i];
            for(int j = 0;j < Width;j++){
                Gradient2[j] += (2.0/DataNum) * gap * ActivedHiddenLayer[j];
            }

            for(int j = 0;j < InNeuron + 1;j++){
                for(int k = 0;k < Width;k++){
                    Gradient1[j][k] = (2.0/DataNum) * gap * ActivedHiddenLayer[k] * (1.0- ActivedHiddenLayer[k]) * ActivedHiddenLayer[k] * InputMatrix[i][j];
                }
            }
            
        }

        for(int i = 0;i < Width;i++){
            WeightMatrix2[i][0] -= LearningRate * Gradient2[i];
        }

        for(int i = 0;i < InNeuron + 1;i++){
            for(int j = 0;j < Width;j++){
                WeightMatrix1[i][j] -= LearningRate * Gradient1[i][j];
            }
        }
        Lost = 0.0;
        for(int i = 0;i < DataNum;i++){
            Lost += (Compute(i) - OutputMatrix[i][0]) * (Compute(i) - OutputMatrix[i][0]);
        }
        Lost /= DataNum;
        printf("%d %lf\n",Train,Lost);
    } while(Lost > e && Train < TrainTime);
}

void writeNeuron()
{
	FILE *fp;
	if((fp=fopen("weight.txt","w"))==NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	for (int i = 0; i < InNeuron + 1;i++){
		for (int j = 0; j < Width; j++){
			fprintf(fp,"%lf ",WeightMatrix1[i][j]);
		}
    }
	fprintf(fp,"\n\n\n\n");

    for(int i = 0;i < Width;i++){
        fprintf(fp,"%lf ",WeightMatrix2[i][0]);
    }

	fclose(fp);
}


double result(double var1,double var2)
{
	int i,j;
	double sum,y;

	var1=(var1-Minin[0]+1)/(Maxin[0]-Minin[0]+1);
	var2=(var2-Minin[1]+1)/(Maxin[1]-Minin[1]+1);

	for (i = 0; i < Width; ++i){
		sum=0;
		sum=WeightMatrix1[1][i]*var1+WeightMatrix1[2][i]*var2 + WeightMatrix1[0][i];
		ActivedHiddenLayer[i]=1/(1+exp(-1*sum));
	}
	sum=0;
	for (j = 0; j < Width; ++j)
		sum+=WeightMatrix2[j][0]*ActivedHiddenLayer[j];

	return sum*(Maxout[0]-Minout[0]+1)+Minout[0]-1;
}