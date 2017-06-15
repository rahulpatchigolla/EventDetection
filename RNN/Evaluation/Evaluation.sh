perl prepare-gold.pl ./Actual_test_events/ ./Prepared_Actual_test_events/
perl prepare-eval.pl -g Prepared_Actual_test_events/ Predicted_test_events/ Prepared_Predicted_test_events/
perl a2-evaluate.pl -g ./Prepared_Actual_test_events/ -spd   ./Prepared_Predicted_test_events/*.a2.t12
