import unittest


class TestBN(unittest.TestCase):

    

    def test_calc_fitness(self):
        # assert that method cal_fitness returns a float
        ea = EA(population_size=5, generations=2, n_hidden_neurons=4)
        self.assertIsInstance(ea.calc_fitness(ea.population[0]), float, "Should be a float")


    def test_evaluate(self):
        # assert that fitness is calculated for every individual
        ea = EA(population_size=5, generations=2, n_hidden_neurons=4)
        self.assertEqual((ea.population_size,), (len(ea.population_fitness),), "[Population_size] fitnesses should be returned")

    def test_generate_offspring(self):
        # assert that the offspring is different from the last
        ea = EA(population_size=5, generations=2, n_hidden_neurons=4)
        offspring = ea.offspring
        ea.generate_offspring()
        self.assertFalse((offspring == ea.offspring).all())

    def test_survivor_selection(self):
        # assert that the population size remains equal after combining offspring and selected parents
        ea = EA(population_size=5, generations=2, n_hidden_neurons=4)
        pop1 = ea.population.shape
        ea.survivor_selection()
        pop2 = ea.population.shape
        self.assertEqual(pop1, pop2, "population size should remain equal after combining")

    def test_check_stagnation(self, stag=10):
        # can encounter an overflow apparently, which is weird
        ea = EA(population_size=5, generations=2, n_hidden_neurons=4)
        l = []
        for i in range(stag):
            l.append(ea.check_stagnation(ea.population_fitness, stag=stag))
        self.assertFalse(all(l[:-1]))
        self.assertTrue(l[-1])

    def test_handle_stagnation(self):
        # assert that the population size remains equal after combining random individuals and selected parents
        ea = EA(population_size=5, generations=2, n_hidden_neurons=4)
        pop1 = ea.population.shape
        ea.handle_stagnation()
        pop2 = ea.population.shape
        self.assertEqual(pop1, pop2)


if __name__ == '__main__':
    unittest.main()
