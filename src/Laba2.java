/**
 * Created by Egor on 12.11.2016.
 */

import java.io.*;
import java.util.*;

public class Laba2 {
    public static void main(String[] args) {
        double buffer;
        FileWriter fileWriter,fileWriter_test;
        String string, s[];
        Perceptrone_model perceptrone_model=new Perceptrone_model(3,5,5,1);

        double data_train[]=new double [200];// ьренировочные данные
        double data_test[]=new double [50];// тестовые
        double data_all[][]=new double[250][2];//все вместе

        int l=0;
        try {
            Scanner scanner = new Scanner(new File("Данные для прогноза.txt"));
            Scanner Scanner_train = new Scanner(new File("train_data.txt"));// и трен и тест уже нормализованы
            Scanner Scanner_test = new Scanner(new File("test_data.txt"));
            while (Scanner_train.hasNextLine()) {
                string = Scanner_train.nextLine();
                s = string.split("\\ ");
                for (int i = 0; i < s.length; i++) {
                    buffer = Double.parseDouble(s[i]);
                    data_train[l]=buffer;
                    l++;
                }
            }
            l=0;

            int k=0;
            while (scanner.hasNextLine()) {
                string = scanner.nextLine();
                s = string.split("\\,");
                for (int i = 0; i < s.length; i++) {
                    if ((i==2)||(i==3)) {
                        buffer = Double.parseDouble(s[i]);
                        data_all[k][l] = buffer;
                        l++;
                    }
                }
                k++;
                l=0;
            }

            double dif_data[]=new double[250];
            for (int i=0;i<k;i++)
                dif_data[i]=data_all[i][0]-data_all[i][1];
//////////////////////Денормализация
            double square_sums=0;
            for (int i=0;i<k;i++)
                square_sums+=dif_data[i]*dif_data[i];
            square_sums=Math.sqrt(square_sums);
            ///////// Конец денормализации

            l=0;


            while (Scanner_test.hasNextLine()) {
                string = Scanner_test.nextLine();
                s = string.split("\\ ");
                for (int i = 0; i < s.length; i++) {
                    buffer = Double.parseDouble(s[i]);
                     data_test[l]=buffer;
                    l++;
                }
            }

            perceptrone_model.prognos_train(data_train,data_test);// учим
            double estimates[];
            estimates=perceptrone_model.get_estimates(data_train,data_test);//прогнозируем

          for (int i=0;i<estimates.length;i++)
                System.out.println("Прогнозируемые данные: "+ estimates[i]*square_sums + " Реальные данные: "+data_train[i]*square_sums);//Домножаем на квадрат тк при нормализации делили
            Scanner_train.close();
            Scanner_test.close();



        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
